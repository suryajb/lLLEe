from comb_utils import *
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# simulation parameters
	parser.add_argument('-Nmodes',type=int,default=2**9,help='number of modes, use a power of 2 for faster simulation')
	parser.add_argument('-total_time',type=float,default=1e-6,help='total time for simulation in seconds, default 1e-6')
	parser.add_argument('-dt',type=float,default=1e-3/2,help='normalized time step (normalized to total loss rate), default 1e-3/2')
	parser.add_argument('-snapshots',type=int,default=2000,help='number of snapshots of the comb solution, default 2000')
	parser.add_argument('-mode_offset',type=int,default=0,help='offset of modes in relation to central pump frequency, this is used for flexibility when Dint curve does not fully fit in range of modes')
	parser.add_argument('-filename',type=str,default='combsol',help="file name for comb solution, default is 'combsol', do not include any suffix (e.g., .csv)")
	parser.add_argument('-save',type=int,default=1,help='indicate if you want to save the comb solution (1 or 0), default is 1 (yes)')
	parser.add_argument('-threads',type=int,default=1,help='indicate the number of threads you want to utilize for FFTs, default is 1')
	parser.add_argument('-plan_fft',type=int,default=1,help='indicate whether you want to use pyfftw, this should be 1 for faster simulation')
	parser.add_argument('-fastmath',type=bool,default=False,help='indicate whether you want to use fastmath in numba, this sacrifices a bit in accuracy for speed')
	parser.add_argument('-numba',type=int,default=1,help='indicate whether you want to use numba for optimized speed')

	# ring parameters
	parser.add_argument('-radius',type=float,default=60e-6,help='radius in meters')
	parser.add_argument('-height',type=float,default=1e-6,help='height of cross section in meters')
	parser.add_argument('-width',type=float,default=2.3e-6,help='width of cross section in meters')
	parser.add_argument('-ng',type=float,default=2.2,help='group velocity index, default 2.2')
	parser.add_argument('-Qc',type=float,default=1e6,help='coupling Q, default 1e6')
	parser.add_argument('-Qi',type=float,default=1e6,help='intrinsic Q, default 1e6')
	parser.add_argument('-n2',type=float,default=2.4e-19,help='kerr nonlinear index, default 2.4e-19')
	parser.add_argument('-Dint_file',type=str,default='2.300w_58.5r_0.960h.csv',help='Dint file, needs to have specific format')
	parser.add_argument('-Dint_degrees',type=int,default=9,help='Degrees to fit the Dint polynomial, default 9')

	# pump laser parameters
	parser.add_argument('-detuning_range', nargs=2, type=int, default=[-8,20], help='detuning range, default [-8,20]')
	parser.add_argument('-frequency_range', nargs=2, type=int, default=[], help='frequency range, default [], use this to replace detuning range')
	parser.add_argument('-wavelength_range', nargs=2, type=int, default=[], help='wavelength range, default [], use this to replace detuning range, prioritizes wavelength -> frequency -> detuning range')
	parser.add_argument('-pump_frequency',type=float,default=299792458/1.550e-6,help='frequency in Hz, default to 1550nm in frequency')
	parser.add_argument('-pump_wavelength',type=float,default=None,help='pump wavelength in meters, default is None, set this to replace pump frequency')
	parser.add_argument('-frequency_sweep_speed',type=float,default=None,help='set laser sweep speed in Hz/sec, default None')
	parser.add_argument('-wavelength_sweep_speed',type=float,default=None,help='set laser sweep speed in nm/sec, default None')
	parser.add_argument('-pump_power',type=float,default=250e-3,help='set laser pump power in watts, default 200e-3')

	args = parser.parse_args()
	
	aln_ring = microring(R=args.radius,height=args.height,width=args.width,ng=args.ng,Qc=args.Qc,Qi=args.Qi,n2=args.n2,
		δnorm_range=args.detuning_range,ω_range=args.frequency_range,λ_range=args.wavelength_range,ω0=args.pump_frequency,
		λ0=args.pump_wavelength,total_time=args.total_time,ω_sweep_speed=args.frequency_sweep_speed,
		λ_sweep_speed=args.wavelength_sweep_speed,Qc_import=10**np.linspace(5.1,7.1,args.Nmodes),
		Dint_file=args.Dint_file,dt=args.dt,
		Nmodes=args.Nmodes,pump=args.pump_power,snapshots=args.snapshots,mode_offset=args.mode_offset,
		Dint_degrees=args.Dint_degrees,save=args.save,filename=args.filename)
	value = input("start simulation? y or n:\n")
	value2 = 'continue'
	idx = None

	if (value == 'y') or (value == 'yes'):
		aln_ring.split_step(plan_fft=args.plan_fft,threads=args.threads,fastmath=args.fastmath,numba_on=args.numba)

		while not(value2 == 'y' or value2 == 'yes'):
			if idx:
				aln_ring.plot_all(int(idx))
			else:
				aln_ring.plot_all()
			value2 = input("quit simulation? y or n:\n")
			if (value2 == 'y' or value2 == 'yes'):
				break
			idx = input('new index for plotting:\n')

	else:
		print('\nsimulation stopped')