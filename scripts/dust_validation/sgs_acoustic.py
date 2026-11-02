"""Measure the fundamental-mode phase of an SGS-supported acoustic eigenmode."""
from common import Report, solver, np, plt


def main():
    report = Report('sgs_acoustic')
    amplitude, gamma, n = 1e-6, 1.4, 64
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    ratios, measured, predicted = [], [], []
    for ratio in (0., .1, .5, 1.):
        s = solver(n=n, dust_on=False, sgs_on=True, sgs_config={'d_model':'none'})
        e0 = ratio/(gamma-1)
        c = np.sqrt(gamma+(10/9)*e0)
        phase = 2*np.pi*s.grid.x
        # Adiabatic SGS compression has gamma_s=5/3. Perturb both pressures to
        # initialize the right-going eigenmode rather than a standing/entropy wave.
        s.U_gas = s.equations.conservative(1+amplitude*np.cos(phase),
                     amplitude*c*np.cos(phase), 1+gamma*amplitude*np.cos(phase))
        s.e_s[:] = e0*(1+(5/3)*amplitude*np.cos(phase))
        initial = s.diagnostics()
        times, mode = [0.], [np.fft.fft(s.U_gas[0]-1)[1]]
        end = 1/c
        while s.t < end-1e-14:
            s.step(min(s.compute_dt(),end-s.t))
            times.append(s.t)
            mode.append(np.fft.fft(s.U_gas[0]-1)[1])
        angle = np.unwrap(np.angle(mode))
        speed = -np.polyfit(times,angle,1)[0]/(2*np.pi)
        error = abs(speed/c-1)
        label = f'e_s/(rho e_int)={ratio:g}'
        # Requested 0.1% accuracy; Fourier spatial derivatives leave finite time
        # integration and frozen-SGS-pressure splitting as the dominant errors.
        report.check(label+' phase speed relative error',error,1e-3)
        report.check(label+' repaired_cells',s.transport_stats['repaired_cells'],0,op='==')
        drift = abs(s.diagnostics()['E_total']/initial['E_total']-1)
        report.check(label+' total energy drift',drift,1e-12)
        report.runs.append(dict(ratio=ratio,N=n,c_measured=speed,c_exact=c,relative_error=error,
             time=s.t,repaired_cells=s.transport_stats['repaired_cells'],
             roundoff_projections=s.transport_stats['roundoff_projections'],
             mode_amplitude_final_over_initial=abs(mode[-1]/mode[0]),energy_drift=drift))
        ratios.append(ratio); measured.append(speed); predicted.append(c)
    ratios, measured, predicted = map(np.asarray, (ratios, measured, predicted))
    # Left: measured error against c_eff, and the error one would make by
    # ignoring the SGS pressure (c = sqrt(gamma p/rho)): the reservoir's
    # mechanical feedback is ~10^3 times larger than the scheme's error.
    err = np.abs(measured/predicted-1)
    no_sgs = np.abs(np.sqrt(gamma)/measured-1)
    axes[0].semilogy(ratios[1:], err[1:], 'o-', label='|c_measured/c_eff - 1|')
    axes[0].semilogy(ratios[1:], no_sgs[1:], 's--', label='|c_gas-only/c_measured - 1|')
    axes[0].axhline(1e-3, color='0.5', ls=':', lw=1, label='0.1% requirement')
    axes[0].set(xlabel='e_s / (rho e_int) [1]', ylabel='relative phase-speed difference',
                title='SGS pressure feedback on acoustics')
    axes[0].legend(fontsize=8)
    axes[1].plot(ratios,predicted,'k-',label='exact c_eff')
    axes[1].plot(ratios,measured,'o',label='measured')
    axes[1].set(xlabel='e_s / (rho e_int) [1]',ylabel='phase speed [L / time]')
    axes[1].legend()
    report.figure(fig,'phase_speed.png')
    return report.finish()


if __name__ == '__main__':
    raise SystemExit(main())
