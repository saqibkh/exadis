#include <iostream>
#include "system.h"
#include "force.h"
#include "mobility.h"
#include "integrator.h"
#include "functions.h"
#include "../debug.h"

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

using namespace ExaDiS;

System* get_system()
{
    // Simulation parameters
    int crystal_type = FCC_CRYSTAL;
    Crystal crystal(crystal_type);
    
    double maxseg = 2000.0;
    double minseg = 300.0;
    std::string datafile = exadis_root_dir + "/examples/22_fcc_Cu_15um_1e3/180chains_16.10e.data";
    SerialDisNet* config = read_paradis(datafile.c_str(), false);
    
    double burgmag = 2.55e-10;
    double MU = 54.6e9;
    double NU = 0.324;
    double a = 6.0;
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    
    return make_system(config, crystal, params);
}

int main(int argc, char* argv[])
{
    ExaDiS::Initialize init(argc, argv);
    System* system = get_system();
    DeviceDisNet* net = system->get_device_network();
    NeighborList* neilist = exadis_new<NeighborList>();
    double cutoff = 1000.0;
    generate_neighbor_list(system, net, neilist, cutoff, Neighbor::NeiSeg);
    int Nsegs = net->Nsegs_local;
    int Ntot = neilist->Ntotnei;
    printf("Nsegs %d Ntotnei %d\n", Nsegs, Ntot);

    // 1) per-pair distance on device (flat range policy)
    Kokkos::View<double*> d_dist("d_dist", Ntot);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, Nsegs), KOKKOS_LAMBDA(const int i) {
        auto count = neilist->get_count();
        auto nei = neilist->get_nei();
        for (int l = 0; l < count[i]; l++) {
            int j = nei(i, l);
            d_dist(neilist->beg(i) + l) = get_min_dist2_segseg(net, i, j);
        }
    });
    Kokkos::fence();
    auto h_dist = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_dist);

    // 2) per-seg team-reduced count (same pattern as BuildSegSegList)
    Kokkos::View<int*> d_cnt("d_cnt", Nsegs);
    Kokkos::View<int*> d_cnt_flat("d_cnt_flat", Nsegs);
    Kokkos::View<int*> d_nei_team("d_nei_team", Nsegs);
    double cutoff2 = cutoff * cutoff;
    typedef Kokkos::TeamPolicy<>::member_type team_handle;
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nsegs, Kokkos::AUTO), KOKKOS_LAMBDA(const team_handle& team) {
        int i = team.league_rank();
        auto count = neilist->get_count();
        auto nei = neilist->get_nei();
        int Nnei = 0;
        Kokkos::single(Kokkos::PerTeam(team), [=] (int& n) { n = count[i]; }, Nnei);
        int nt;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, Nnei), [&] (const int& l, int& nsum) {
            int j = nei(i, l);
            if (i < j) {
                double dist2 = get_min_dist2_segseg(net, i, j);
                if (dist2 >= 0.0 && dist2 < cutoff2) nsum++;
            }
        }, nt);
        Kokkos::single(Kokkos::PerTeam(team), [=] () { d_cnt(i) = nt; d_nei_team(i) = Nnei; });
    });
    Kokkos::fence();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, Nsegs), KOKKOS_LAMBDA(const int i) {
        auto count = neilist->get_count();
        auto nei = neilist->get_nei();
        int n = 0;
        for (int l = 0; l < count[i]; l++) {
            int j = nei(i, l);
            if (i < j) {
                double dist2 = get_min_dist2_segseg(net, i, j);
                if (dist2 >= 0.0 && dist2 < cutoff2) n++;
            }
        }
        d_cnt_flat(i) = n;
    });
    Kokkos::fence();
    auto h_cnt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_cnt);
    auto h_cnt_flat = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_cnt_flat);
    auto h_nei_team = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_nei_team);

    // host reference
    SerialDisNet* snet = system->get_serial_network();
    int ndist = 0, nteam = 0, nflat = 0, nneiteam = 0, shown = 0;
    long tot_host = 0, tot_team = 0, tot_flat = 0;
    for (int i = 0; i < Nsegs; i++) {
        int c = neilist->count(i), b = neilist->beg(i), href = 0;
        for (int l = 0; l < c; l++) {
            int j = neilist->list(b + l);
            double hd = get_min_dist2_segseg(snet, i, j);
            double dd = h_dist(b + l);
            if (!(fabs(hd - dd) <= 1e-9 * fmax(1.0, fabs(hd)))) {
                ndist++;
                if (shown++ < 15) printf("dist mismatch i=%d j=%d host=%.17g dev=%.17g\n", i, j, hd, dd);
            }
            if (i < j && hd >= 0.0 && hd < cutoff2) href++;
        }
        tot_host += href; tot_team += h_cnt(i); tot_flat += h_cnt_flat(i);
        if (h_cnt(i) != href) { nteam++; if (nteam <= 10) printf("team count mismatch i=%d host=%d team=%d flat=%d nnei=%d teamnnei=%d\n", i, href, h_cnt(i), h_cnt_flat(i), c, h_nei_team(i)); }
        if (h_cnt_flat(i) != href) nflat++;
        if (h_nei_team(i) != c) nneiteam++;
    }

    // 3) stage split: device pbc_position vs host, and device get_min_dist2 on host-made r's
    Kokkos::View<double**> h_in("h_in", Ntot, 12);   // host r1..r4
    auto hh_in = Kokkos::create_mirror_view(Kokkos::HostSpace(), h_in);
    FILE* fd = fopen("pairs_dump.txt", "w");
    for (int i = 0; i < Nsegs; i++) {
        int c = neilist->count(i), b = neilist->beg(i);
        for (int l = 0; l < c; l++) {
            int j = neilist->list(b + l);
            auto nodes = snet->get_nodes(); auto segs = snet->get_segs(); auto cell = snet->cell;
            Vec3 r1 = nodes[segs[i].n1].pos;
            Vec3 r2 = cell.pbc_position(r1, nodes[segs[i].n2].pos);
            Vec3 r3 = cell.pbc_position(r1, nodes[segs[j].n1].pos);
            Vec3 r4 = cell.pbc_position(r3, nodes[segs[j].n2].pos);
            Vec3 rr[4] = {r1, r2, r3, r4};
            for (int k = 0; k < 4; k++) { hh_in(b+l, 3*k) = rr[k].x; hh_in(b+l, 3*k+1) = rr[k].y; hh_in(b+l, 3*k+2) = rr[k].z; }
            if (i < 40) fprintf(fd, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z, r4.x, r4.y, r4.z, get_min_dist2(r1, r2, r3, r4));
        }
    }
    fclose(fd);
    Kokkos::deep_copy(h_in, hh_in);
    Kokkos::View<double**> d_out("d_out", Ntot, 13);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, Nsegs), KOKKOS_LAMBDA(const int i) {
        auto count = neilist->get_count();
        auto nei = neilist->get_nei();
        auto nodes = net->get_nodes(); auto segs = net->get_segs(); auto cell = net->cell;
        for (int l = 0; l < count[i]; l++) {
            int j = nei(i, l); int p = neilist->beg(i) + l;
            Vec3 r1 = nodes[segs[i].n1].pos;
            Vec3 r2 = cell.pbc_position(r1, nodes[segs[i].n2].pos);
            Vec3 r3 = cell.pbc_position(r1, nodes[segs[j].n1].pos);
            Vec3 r4 = cell.pbc_position(r3, nodes[segs[j].n2].pos);
            Vec3 rr[4] = {r1, r2, r3, r4};
            for (int k = 0; k < 4; k++) { d_out(p, 3*k) = rr[k].x; d_out(p, 3*k+1) = rr[k].y; d_out(p, 3*k+2) = rr[k].z; }
            Vec3 a(h_in(p,0),h_in(p,1),h_in(p,2)), bb(h_in(p,3),h_in(p,4),h_in(p,5)), cc(h_in(p,6),h_in(p,7),h_in(p,8)), dd(h_in(p,9),h_in(p,10),h_in(p,11));
            d_out(p, 12) = get_min_dist2(a, bb, cc, dd);
        }
    });
    Kokkos::fence();
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);
    int npbc = 0, nmd = 0; shown = 0;
    for (int p = 0; p < Ntot; p++) {
        bool bad = false;
        for (int k = 0; k < 12; k++) if (h_out(p,k) != hh_in(p,k)) bad = true;
        if (bad) { npbc++; if (shown++ < 5) printf("pbc mismatch p=%d host r2=(%.17g %.17g %.17g) dev r2=(%.17g %.17g %.17g)\n", p, hh_in(p,3), hh_in(p,4), hh_in(p,5), h_out(p,3), h_out(p,4), h_out(p,5)); }
        Vec3 a(hh_in(p,0),hh_in(p,1),hh_in(p,2)), bb(hh_in(p,3),hh_in(p,4),hh_in(p,5)), cc(hh_in(p,6),hh_in(p,7),hh_in(p,8)), dd(hh_in(p,9),hh_in(p,10),hh_in(p,11));
        double hd = get_min_dist2(a, bb, cc, dd);
        if (!(fabs(hd - h_out(p,12)) <= 1e-9 * fmax(1.0, fabs(hd)))) { nmd++; if (nmd <= 5) printf("min_dist2 mismatch p=%d host=%.17g dev=%.17g in=%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", p, hd, h_out(p,12), a.x,a.y,a.z,bb.x,bb.y,bb.z,cc.x,cc.y,cc.z,dd.x,dd.y,dd.z); }
    }
    printf("pbc mismatches %d, get_min_dist2 mismatches %d (of %d)\n", npbc, nmd, Ntot);
    printf("dist mismatches %d of %d\n", ndist, Ntot);
    printf("team count mismatches %d, flat count mismatches %d, team Nnei mismatches %d (of %d segs)\n", nteam, nflat, nneiteam, Nsegs);
    printf("totals host %ld team %ld flat %ld\n", tot_host, tot_team, tot_flat);
    exadis_delete(neilist);
    exadis_delete(system);
    return 0;
}
