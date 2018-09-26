import numpy as np
import healpy as hp
import pyfits
import os
import math
import sys

def tag_exist(hdu,col): #test if fits hdu has column col
    if type(col) is not list:
        coll = [col]
    else:
        coll = col
    haveTag = []
    for cn in coll:
        if cn.upper() in hdu.columns.names or cn in hdu.columns.names:
            haveTag.append(1)
        else:
            haveTag.append(0)
    return np.array(haveTag)

def pos_rot(px,py,pz,rmat):
    rpx = px*rmat[0,0] + py*rmat[0,1] + pz*rmat[0,2]
    rpy = px*rmat[1,0] + py*rmat[1,1] + pz*rmat[1,2]
    rpz = px*rmat[2,0] + py*rmat[2,1] + pz*rmat[2,2]
    return (rpx,rpy,rpz)

def radec_rot(ra,dec,rmat):
    d2r = np.pi/180.0
    px = np.cos(ra*d2r)*np.cos(dec*d2r)
    py = np.sin(ra*d2r)*np.cos(dec*d2r)
    pz = np.sin(dec*d2r)

    (rpx,rpy,rpz) = pos_rot(px,py,pz,rmat)

    rra = np.arctan2(rpy,rpx)/d2r
    rdec = 90.0 - np.arccos(rpz/np.sqrt(rpx*rpx + rpy*rpy + rpz*rpz))/d2r

    #remap ra to [0,360)
    for x in np.nditer(rra, op_flags=['readwrite']):
        if (x < 0.0):
            x += 360.0
        if (x>= 360.0):
            x -= 360.0

    return (rra,rdec)

def ttens_rot(a00,a01,a10,a11,ra,dec,rmat):
    #assumes tensor is in theta-phi basis!

    #assumes ra-dec in decimal degrees
    d2r = math.pi/180.0
    r2d = 180.0/math.pi
    px = np.cos(ra*d2r)*np.cos(dec*d2r)
    py = np.sin(ra*d2r)*np.cos(dec*d2r)
    pz = np.sin(dec*d2r)

    #get rotated vec
    (rpx,rpy,rpz) = pos_rot(px,py,pz,rmat)

    #get rotated ephi vec
    ephi_px = -py.copy()
    ephi_py = px.copy()
    ephi_pz = pz.copy()
    ephi_pz[:] = 0.0
    (rephi_px,rephi_py,rephi_pz) = pos_rot(ephi_px,ephi_py,ephi_pz,rmat)

    #get ephi,etheta of rotated vec
    ephi_rpx = -rpy.copy()
    ephi_rpy = rpx.copy()
    ephi_rpz = rpz.copy()
    ephi_rpz[:] = 0.0
    etheta_rpx = rpz*rpx
    etheta_rpy = rpz*rpy
    etheta_rpz = -1.0*(rpx*rpx + rpy*rpy)

    #now comp para. ang
    norm = np.sqrt((1.0 - rpz)*(1.0 + rpz)*(1.0 - pz)*(1.0 + pz))
    sinpsi = (rephi_px*etheta_rpx + rephi_py*etheta_rpy + rephi_pz*etheta_rpz)/norm
    cospsi = (rephi_px*ephi_rpx + rephi_py*ephi_rpy + rephi_pz*ephi_rpz)/norm


    #now do rots: Ar = R.A.Rt
    t00 = a00*cospsi + a01*sinpsi
    t01 = a00*(-sinpsi) + a01*cospsi
    t10 = a10*cospsi + a11*sinpsi
    t11 = a10*(-sinpsi) + a11*cospsi

    ra00 = cospsi*t00 + sinpsi*t10
    ra01 = cospsi*t01 + sinpsi*t11
    ra10 = -sinpsi*t00 + cospsi*t10
    ra11 = -sinpsi*t01 + cospsi*t11

    return (ra00,ra01,ra10,ra11)

def epsgam_rot(g1,g2,ra,dec,rmat):
    #first covnert eps to theta-phi
    rda00 = 1.0 - g1.copy()
    rda01 = -g2.copy()
    rda10 = -g2.copy()
    rda11 = 1.0 + g1.copy()
    a00 = rda11.copy()
    a01 = -rda10.copy()
    a10 = -rda01.copy()
    a11 = rda00.copy()

    #do rot
    (rda00,rda01,rda10,rda11) = ttens_rot(a00,a01,a10,a11,ra,dec,rmat)

    #convert back to ra-dec
    a00 = rda11
    a01 = -rda10
    a10 = -rda01
    a11 = rda00

    #get epslion
    re1 = (a11 - a00)/2.0
    re2 = -0.5*(a01+a10)

    return (re1,re2)

def rot_mock_file(fname,rmat,nfname,footprint=None,nside=None,nest=False):
    #get file
    print( "Reading: '%s'" % fname)
    sys.stdout.flush()
    hdu = pyfits.open(fname)
    d = hdu[1].data

    #get file gaain for rotated version
    nhdu = pyfits.open(fname)
    nd = nhdu[1].data

    didRot = False
    badRot = False

    #do positions & vels
    te = tag_exist(d,['px','py','pz'])
    if te.sum() == 3:
        (rpx,rpy,rpz) = pos_rot(d['px'],d['py'],d['pz'],rmat)
        nd['px'][:] = rpx[:]
        nd['py'][:] = rpy[:]
        nd['pz'][:] = rpz[:]
        didRot = True
    elif te.sum() > 0:
        print( "Error: File '%s' has weird P[X,Y,Z] tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    if footprint is not None:
        pix = hp.vec2pix(nside,nd['px'],nd['py'],nd['pz'], nest=nest)
        guse = np.in1d(pix, footprint['HPIX'])
        if not any(guse):
            print("No galaxies in this pixel fall within the footprint")
            return

    te = tag_exist(d,['halopx','halopy','halopz'])
    if te.sum() == 3:
        (rpx,rpy,rpz) = pos_rot(d['halopx'],d['halopy'],d['halopz'],rmat)
        nd['halopx'][:] = rpx[:]
        nd['halopy'][:] = rpy[:]
        nd['halopz'][:] = rpz[:]
        didRot = True
    elif te.sum() > 0:
        print( "Error: File '%s' has weird HALOP[X,Y,Z] tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    te = tag_exist(d,['vx','vy','vz'])
    if te.sum() == 3:
        (rpx,rpy,rpz) = pos_rot(d['vx'],d['vy'],d['vz'],rmat)
        nd['vx'][:] = rpx[:]
        nd['vy'][:] = rpy[:]
        nd['vz'][:] = rpz[:]
        didRot = True
    elif te.sum() > 0:
        print( "Error: File '%s' has weird V[X,Y,Z] tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    te = tag_exist(d,['halovx','halovy','halovz'])
    if te.sum() == 3:
        (rpx,rpy,rpz) = pos_rot(d['halovx'],d['halovy'],d['halovz'],rmat)
        nd['halovx'][:] = rpx[:]
        nd['halovy'][:] = rpy[:]
        nd['halovz'][:] = rpz[:]
        didRot = True
    elif te.sum() > 0:
        print( "Error: File '%s' has weird HALOV[X,Y,Z] tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    #do ra-dec
    te = tag_exist(d,['ra','dec'])
    if te.sum() == 2:
        (rra,rdec) = radec_rot(d['ra'],d['dec'],rmat)
        nd['ra'][:] = rra[:]
        nd['dec'][:] = rdec[:]
        didRot = True
    elif te.sum() > 0:
        print( "Error: File '%s' has weird RA,DEC tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    te = tag_exist(d,['tra','tdec'])
    if te.sum() == 2:
        (rra,rdec) = radec_rot(d['tra'],d['tdec'],rmat)
        nd['tra'][:] = rra[:]
        nd['tdec'][:] = rdec[:]
        didRot = True
    elif te.sum() > 0:
        print( "Error: File '%s' has weird TRA,TDEC tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    #do shear
    te = tag_exist(d,['gamma1','gamma2','ra','dec'])
    if te.sum() == 4:
        (rg1,rg2) = epsgam_rot(d['gamma1'],d['gamma2'],d['ra'],d['dec'],rmat)
        nd['gamma1'][:] = rg1[:]
        nd['gamma2'][:] = rg2[:]
        didRot = True
    elif te.sum() > 0 and te[0] == 1 and te[1] == 1:
        print( "Error: File '%s' has weird GAMMA[1,2],RA,DEC tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    te = tag_exist(d,['te','ra','dec'])
    if te.sum() == 3:
        (rg1,rg2) = epsgam_rot(d['te'][:,0],d['te'][:,1],d['ra'],d['dec'],rmat)
        nd['te'][:,0] = rg1[:]
        nd['te'][:,1] = rg2[:]
        didRot = True
    elif te.sum() > 0 and te[0] == 1:
        print( "Error: File '%s' has weird TE,RA,DEC tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    te = tag_exist(d,['epsilon','ra','dec'])
    if te.sum() == 3:
        (rg1,rg2) = epsgam_rot(d['epsilon'][:,0],d['epsilon'][:,1],d['ra'],d['dec'],rmat)
        nd['epsilon'][:,0] = rg1[:]
        nd['epsilon'][:,1] = rg2[:]
        didRot = True
    elif te.sum() > 0 and te[0] == 1:
        print( "Error: File '%s' has weird EPSILON,RA,DEC tags!" % fname,te)
        sys.stdout.flush()
        badRot = True

    #write the file
    if didRot and not badRot:
        print( "Writing: '%s'" % nfname)
        sys.stdout.flush()
        try:
            os.remove(nfname)
        except OSError:
            pass

        nhdu.writeto(nfname)

    return nd
