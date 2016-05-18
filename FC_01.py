import vapoursynth as vs
import functools
import havsfunc as haf

core = vs.get_core()

#Conditional Decomb Function:
def conditionalDecomb(n, f, orig, decomb):
    if f.props._Combed:
        return decomb
    else:
        return orig
        
#Turning Functions:
def TurnLeft(input, reset=False):
	turn = core.std.Transpose(input).std.FlipVertical()
	
	if reset is True:
		reset = core.std.Transpose(turn).std.FlipHorizontal()
		
		return reset
		
	else:
		return turn
		
def TurnRight(input, reset=False):
	turn = core.std.Transpose(input).std.FlipHorizontal()
	
	if reset is True:
		reset = core.std.Transpose(turn).std.FlipVertical()
		
		return reset
		
	else:
		return turn
      
#Anti-Aliasing Function:
def taa(input, aa=48):
	c = core.fmtc.resample(input, w=input.width*2, h=input.height*2, kernel='spline36').fmtc.bitdepth(
			bits=8, dmode=1)
	
	r = TurnRight(c).sangnom.SangNomMod(
			aa=aa)
	l = TurnLeft(TurnRight(r, reset=True)).sangnom.SangNomMod(
			aa=aa)
	
	o = core.fmtc.bitdepth(TurnLeft(l, reset=True), bits=16).fmtc.resample(
			w=input.width, h=input.height, kernel='bicubic', a1=0, a2=0.5)
	
	return o

#Line Mask:
def l_msk(input, min, max):
	msk = core.generic.Prewitt(input, min=min, max=max)
	o = core.rgvs.RemoveGrain(msk, mode=20)

	return o

#Detail/Gradiant Mask:
def d_msk(input, min):
	msk = core.generic.TEdge(input, min=min).rgvs.RemoveGrain(mode=20)
	o = core.generic.Invert(msk)
	
	return msk

#Halo Mask:
def h_msk(input, th):
	msk = core.generic.Canny(input, sigma=2, th=th)
	o = core.generic.Inflate(msk).generic.Inflate().rgvs.RemoveGrain(mode=20)
	
	return o

#Source Filter:
v = core.d2v.Source(input=r'~\Fantastic Children\raw\Indexed\01\FC_01.d2v')
v = core.std.Trim(v, 0, 43469)

#IVTC:
deint = core.vivtc.VFM(v, order=1, field=1, mode=5)
decomb = core.tdm.TDeintMod(deint, order=1, mtype=2,
	edeint=core.nnedi3.nnedi3(v, field=1))
	
combed = core.tdm.IsCombed(deint, cthresh=16)
v = core.std.FrameEval(v, functools.partial(conditionalDecomb, orig=deint, decomb=decomb),
		combed)
		
v = core.vivtc.VDecimate(v)
v = core.fmtc.resample(v, w=640, h=480, kernel='sinc')

#Upsample:
v = core.fmtc.bitdepth(v, bits=16)

#Temp-Denoise:
v = haf.SMDegrain(v, tr=3, thSAD=200, RefineMotion=True, prefilter=1, Str=1.2, 
		contrasharp=True, pel=4, subpixel=3)

#Anti-Aliasing:
aa_msk = l_msk(v, min=4000, max=6000)
aa = taa(v, aa=48)
v = core.std.MaskedMerge(v, aa, mask=aa_msk)

#Spatial-Denoise:
dft_msk = d_msk(v, min=7500)
dft = core.dfttest.DFTTest(v, sigma=4)
v = core.std.MaskedMerge(v, dft, mask=dft_msk)

#Upscale:
v = core.nnedi3.nnedi3_rpow2(v, rfactor=2)

#De-Halo:
halo_msk = h_msk(v, th=1000)
dehalo = haf.DeHalo_alpha(v, rx=2, ry=2, darkstr=0.5, brightstr=1.5)
v = core.std.MaskedMerge(v, dehalo, mask=halo_msk)

#Dither:
dith_msk = d_msk(v, min=12500)
dith = core.f3kdb.Deband(v, grainy=64, grainc=64, dither_algo=3, output_depth=16)
v = core.std.MaskedMerge(v, dith, mask=dith_msk)

#Downsample:
v = core.fmtc.bitdepth(v, bits=10, dmode=6)

#Output:
v.set_output()
