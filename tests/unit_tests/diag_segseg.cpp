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
    printf("dist mismatches %d of %d\n", ndist, Ntot);
    printf("team count mismatches %d, flat count mismatches %d, team Nnei mismatches %d (of %d segs)\n", nteam, nflat, nneiteam, Nsegs);
    printf("totals host %ld team %ld flat %ld\n", tot_host, tot_team, tot_flat);
    exadis_delete(neilist);
    exadis_delete(system);
    return 0;
}
