#This software is distributed without amy warranty 

#Please, include the reference to the paper below in your publications if you use this software:
#De Vicente, J.; Sanchez, E.; Sevilla-Noarbe, I.,"DNF - Galaxy photometric redshift by Directional Neighbourhood Fitting", MNRAS, Vol. 459, Issue 3, pag 3078-3088, 2016.

#!/usr/bin/python
"""
nf module: contains two alternative functions to compute the photoz of a photometric sample of galaxies.
enf: Euclidean Neighborhood Fit (faster) 
dnf: Directional Neighborhood Fit (better)

The following python libraries are used in the code:
import math
import numpy
from sklearn import neighbors  
"""
__author__ = "Juan de Vicente"
__copyright__ = "Copyright 2015, Juan de Vicente"
__version__ = "4.0.4"
__email__= "juan.vicente@ciemat.es"

import math
import numpy as np
#from numba import jit
from sklearn import neighbors
from time import time
import sys


def enf(T,z,V,Verr,zbins,pdf=True,bound=True,Nneighbors=40):
    """
    Computes the photo-z by Euclidean Neighborhood Fit (Copyright (C) 2015, Juan de Vicente)
  
    Input parameters:
      T: 2-dimensional array with fluxes/magnitudes of the training sample
      z: 1-dimensional array with the spectroscopic redshift of the training sample
      V: 2-dimensional array with fluxes/magnitudes of the photometric sample
      zbins: 1-dimensional numpy array with redshift bins for photo-z PDFs
      pdf: True for pdf computation
      bound: ensure photo-z remain within training redshift range 
      Nneighbors: number of neighbors
    Return:
      photoz: 1-dimesional photo-z array for the photometric sample
      photoz_err: 1-dimensional photo-z error estimation array
      pdf: 2-dimensional photo-z PDFs array when pdf==1, 0 when pdf==0
      z1: for N(z) histogramming. When working with bins, use photoz for classification in bins and z1 for n(z)
      nneighbors: number of neighbors
      radius: radius of the hypersphere containing all neighbors
    """      
    

    
    nfilters=T.shape[1]
    Nvalid=V.shape[0]
    knnz=np.zeros(Nvalid,dtype='double')
    photoz=np.zeros(Nvalid,dtype='double')
    photozerr=np.zeros(Nvalid,dtype='double')
    z1=np.zeros(Nvalid,dtype='double')
    radius=np.zeros(Nvalid,dtype='double')
    nneighbors=np.zeros(Nvalid,dtype='double')
    
    A=np.zeros((Nneighbors,nfilters),dtype='double')
    B=np.zeros(Nneighbors,dtype='double')
    maxz=np.max(z)
    minz=np.min(z)
    ########bins##########
    nbins=len(zbins)-1 
    bincenter=(np.double(zbins[1:])+np.double(zbins[:-1]))/2.0
    if pdf==1:
        Vpdf=np.zeros((Nvalid,nbins),dtype='double')
    
    

        

    scale=10000.0

    zt_true=np.int_(scale*z) 
    clf=neighbors.KNeighborsRegressor(n_neighbors=Nneighbors)
    clf.fit(T[::2],zt_true)  #multimagnitude-redshift association from the training sample
    photoz=clf.predict(V)
    photoz=np.double(photoz)/scale
    Vdistances,Vneighbors= clf.kneighbors(V,n_neighbors=Nneighbors)  #neighbors computation

    
    ####Neighborhood fit
    for i in range(0,Nvalid):  
        Nneighbors=Vneighbors[i].shape[0]
        NEIGHBORS=np.zeros(Nneighbors,dtype=[('pos','i4'),('distance','f8'),('z_true','f8')])
        
        NEIGHBORS['distance']=Vdistances[i]         
        NEIGHBORS['z_true']=z[Vneighbors[i]]
        NEIGHBORS['pos']=Vneighbors[i]      
            
        z1[i]=NEIGHBORS[0]['z_true']

        if NEIGHBORS[0]['distance']==0.0:  
            photoz[i]=NEIGHBORS[0]['z_true']
    
            photozerr[i]=NEIGHBORS['z_true'].std()
            if pdf==1:
                zdist=photoz[i] #-0.01 #-residuals  #for pdf
                hist=np.double(np.histogram(zdist,zbins)[0])
                Vpdf[i]=hist/np.sum(hist)
            
            continue
                    
        #neighbor selection
        NEIGHBORS=NEIGHBORS[0:Nneighbors]  #from 1 in case to exclude the own galaxy
        neigh=Nneighbors
        radius[i]=NEIGHBORS[neigh-1]['distance']
        
        nneighbors[i]=neigh
         
        fititerations=4
        for h in range(0,fititerations):
            
            A=T[NEIGHBORS['pos']]  
            B=z[NEIGHBORS['pos']]
        
            X=np.linalg.lstsq(A,B)
            residuals=B-np.dot(A,X[0])
        
            if h==0:
                neig=NEIGHBORS.shape[0]
                photoz[i]=np.inner(X[0],V[i])
                if X[1].size!=0:
                    err1=np.inner(np.abs(X[0]),Verr[i])
                    err2=0
                    photozerr[i]=np.sqrt(err1*err1+err2*err2)
                else:
                    photozerr[i]=0.01
                
                if pdf==1:  #PDFs computation 
                    zdist=photoz[i]+residuals
                    hist=np.double(np.histogram(zdist,zbins)[0])
                    sumhist=np.sum(hist)
                    if sumhist==0.0:
                        Vpdf[i][:]=0.0
                    else:
                        Vpdf[i]=hist/sumhist 
                else:
                    Vpdf=0
    
            absresiduals=np.abs(residuals)   #outlayers are removed after each iteration 
            sigma3=3.0*np.std(residuals)
            selection=(absresiduals<sigma3)
            
            NEIGHBORS=NEIGHBORS[selection]
           

                
        
        photoz[i]=np.inner(X[0],V[i])

        if bound==1:
            if photoz[i]< minz or photoz[i]>maxz:
               photozerr[i]+=np.abs(photoz[i]-NEIGHBORS[0]['z_true'])
                                  
               photoz[i]=NEIGHBORS[0]['z_true']

        percent=np.double(100*i)/Nvalid
        
        if i % 1000 ==1:
            print('progress: ',percent,'%')
    
    
    return photoz,photozerr,Vpdf,z1,nneighbors,radius


#@jit
def dnf(T,z,V,Verr,zbins,pdf=True,bound=False,radius=2.0,Nneighbors=80,magflux='flux'):
    """
    def dnf(T,z,V,Verr,zbins,pdf=True,bound=False,radius=2.0,Nneighbors=80,magflux='flux')
    
    Computes the photo-z by Directional Neighborhood Fit (Copyright (C) 2015, Juan de Vicente)
  
    Input parameters:
      T: 2-dimensional array with fluxes/magnitudes of the training sample
      z: 1-dimensional array with the spectroscopic redshift of the training sample
      V: 2-dimensional array with fluxes/magnitudes of the photometric sample
      Verr: 2-dimensional array with magnitudes errors of the photometric sample
      zbins: 1-dimensional numpy array with redshift bins for photo-z PDFs
      pdf: Let be True for pdf computation
      bound: True to ensure photometric redshifts remain inside the training redshift range.
      radius: Euclidean-radius for euclidean neighbors preselection to speed up and avoid outliers.
      Galaxies without neighbors inside this radius are tagged with photoz_err=99.0 and should be removed from statistical analysis.
      Nneighbors: Number of neighbors to construct the photo-z hyperplane predictor (number of neighbors for the fit)
      magflux: 'mag' | 'flux'
    Return:
      photoz: 1-dimesional dnf-photoz array for the photometric sample
      photoz_err: 1-dimensional photoz error estimation array. Takes the value 99.0 for galaxies with unreliable photo-z
      Vpdf: 2-dimensional photo-z PDFs array when pdf==1, 0 when pdf==0
      z1: 1-dimesional photo-z array to be used for N(z) histogramming. When computing n(z) per bin, use dnf-photoz for galaxy classification in bins and z1 for n(z) histogramming.
      nneighbors: 1-dimensional array with the number of neighbors used in the photo-z estimation for each galaxy
      closestNeighborDistance: 1-dimensional array with the Euclidean-distance of the closest neighbor for each galaxy
    """      

    
    nfilters=T.shape[1]
    Nvalid=V.shape[0]
    Ntrain=T.shape[0]
    
     #output declaration
    photoz=np.zeros(Nvalid,dtype='double')
    z1=np.zeros(Nvalid,dtype='double')  
    photozerr=np.zeros(Nvalid,dtype='double')
    nneighbors=np.zeros(Nvalid,dtype='double')

    #conversion to fluxes
    if magflux=='mag': 
          for j in range(nfilters):
             T[:,j]=np.power(10.0,-T[:,j]/2.5)  
             V[:,j]=np.power(10.0,-V[:,j]/2.5)             
             Verr[:,j]=V[:,j]*Verr[:,j]/1.08        
    
    #neighbor preselection with euclidean radius
    if Ntrain<2000:
        Nneighborspre=Ntrain
    else:
        Nneighborspre=2000  

    start = time()
    clf=neighbors.KNeighborsRegressor(n_neighbors=Nneighborspre, n_jobs=4)
    clf.fit(T,z)  #multimagnitude-redshift association from the training sample
    end = time()
    print('Fitting kneighbors took: {}s'.format(end - start))
    
    #photoz=clf.predict(V)
    start = time()
    Vdistances,Vneighbors= clf.kneighbors(V,n_neighbors=Nneighborspre)  #neighbors computation
    end = time()
    print('Finding euclidean neighbors took {}s'.format(end - start))

    closestNeighborDistance=Vdistances[:,0]
    Vclosest=Vneighbors[:,0]
    
    #auxiliary variable declaration
    pescalar=np.zeros(Ntrain,dtype='double')
    D2=np.zeros(Ntrain,dtype='double')
    Tnorm=np.zeros(Ntrain,dtype='double')
    Tnorm2=np.zeros(Ntrain,dtype='double')    
    #max and min training photo-zs
    maxz=np.max(z)
    minz=np.min(z)

    ########bins##########
    nbins=len(zbins)-1 
    bincenter=(np.double(zbins[1:])+np.double(zbins[:-1]))/2.0
    if pdf==True:
     Vpdf=np.zeros((Nvalid,nbins),dtype='double')

    #Normas
    for t,i in zip(T,list(range(Ntrain))):
     Tnorm[i]=np.linalg.norm(t)
     Tnorm2[i]=np.inner(t,t)

    #for offset of the fit
    Te=np.ones((Ntrain,nfilters+1),dtype='double')  
    Te[:,:-1]=T
    Ve=np.ones((Nvalid,nfilters+1),dtype='double')  
    Ve[:,:-1]=V


    coef=np.power(10,radius/2.5) #conversion of mag radius to flux 
    #photoz computation
    for i in range(0,Nvalid):
        #PRESELECTION ON Euclidean flux radius
        selection=np.ones(Nneighborspre,dtype='bool') 
        for j in range(0,nfilters):
                  selectionaux=np.logical_and(V[i][j]/T[Vneighbors[i],j]<coef,T[Vneighbors[i],j]/V[i][j]<coef)
                  selection=np.logical_and(selection,selectionaux)
           
        Vneighbo=Vneighbors[i][selection]
        Vdistanc=Vdistances[i][selection]
    
        Eneighbors=Vneighbo.size #euclidean neighbors within radius R
    
        if Eneighbors==0:  #probably bad photo-zs
            nneighbors[i]=0
            photozerr[i]=99.0
            photoz[i]=z[Vclosest[i]]
            continue

        #copy of the previous euclidean neighbors computed
        NEIGHBORS=np.zeros(Eneighbors,dtype=[('pos','i4'),('distance','f8'),('z_true','f8')])
        Ts=T[Vneighbo]
        zs=z[Vneighbo]
        Tsnorm=Tnorm[Vneighbo]
        Tsnorm2=Tnorm2[Vneighbo]
        
        #metric computation
        D=V[i]-Ts
        Dsquare=D*D
        D2=np.sum(Dsquare,axis=1)

        Vnorm=np.linalg.norm(V[i])
        pescalar=np.inner(V[i],Ts)
        normalization=Vnorm*Tsnorm
        NIP=pescalar/normalization
         
        NEIGHBORS['distance']=np.arccos(NIP)*np.sqrt(D2)
        NEIGHBORS['z_true']=zs
        NEIGHBORS[:]['pos']=Vneighbo      
        NEIGHBORSsort=np.sort(NEIGHBORS,order='distance')
        z1[i]=NEIGHBORSsort[0]['z_true']
        
        if NEIGHBORSsort[0]['distance']==0.0:
            photoz[i]=NEIGHBORSsort[0]['z_true']
            if nneighbors[i]==0:
                photozerr[i]=0.001
            else:
                photozerr[i]=NEIGHBORSsort['z_true'].std()

            if pdf==True:
                zdist=photoz[i] #-0.01 #-residuals  #for p
                hist=np.double(np.histogram(zdist,zbins)[0])
                Vpdf[i]=hist/np.sum(hist)
            continue
        
        #neighbor selection
        if Eneighbors>Nneighbors:
                NEIGHBORSsort=NEIGHBORSsort[0:Nneighbors]  #from 1 in case to exclude the own galaxy
                neigh=Nneighbors
        else:
                neigh=Eneighbors
        
        nneighbors[i]=neigh
  
            
        #nearest neighbor photo-z computation when few neighbors are found
        if neigh<30: 
            photoz[i]=np.inner(NEIGHBORSsort['z_true'],1.0/NEIGHBORSsort['distance'])/np.sum(1.0/NEIGHBORSsort['distance']) 
            if neigh==1:
                photozerr[i]=0.1
            else:
                photozerr[i]=np.std(NEIGHBORSsort['z_true'])
            
            if pdf==True:
                        if photozerr[i]==0:
                            s=1
                        else:
                            s=photozerr[i]
                        zdist=np.random.normal(photoz[i],s,neigh)
                        #zdist=NEIGHBORSsort['z_true']
                        hist=np.double(np.histogram(zdist,zbins)[0])
                        sumhist=np.sum(hist)
                        if sumhist==0.0:
                            Vpdf[i][:]=0.0
                        else:
                            Vpdf[i]=hist/sumhist 
            else:
                        Vpdf[i]=0
                      
            continue

   
        #Fit    
        fititerations=4
        for h in range(0,fititerations):
            A=Te[NEIGHBORSsort['pos']]  
            B=z[NEIGHBORSsort['pos']]
            X=np.linalg.lstsq(A,B)
            residuals=B-np.dot(A,X[0])
           
            if h==0:
            #PDFs computation
                photoz[i]=np.inner(X[0],Ve[i])
            
                if pdf==True:
                    zdist=photoz[i]+residuals
                    hist=np.double(np.histogram(zdist,zbins)[0])
                    sumhist=np.sum(hist)
                    if sumhist==0.0:
                        Vpdf[i][:]=0.0
                    else:
                        Vpdf[i]=hist/sumhist 
                else:
                    Vpdf=0
                
                #error estimation
                neig=NEIGHBORSsort.shape[0]
                if X[1].size!=0:
                    err1=np.inner(np.abs(X[0][:-1]),Verr[i])
                    #err2=np.sqrt(X[1]/neig)
                    err2=0
                    photozerr[i]=np.sqrt(err1*err1+err2*err2)
                else:
                    photozerr[i]=0.01
            
            absresiduals=np.abs(residuals)       #outlayers are removed after each iteration 
            sigma3=3.0*np.mean(absresiduals)
            selection=(absresiduals<sigma3)
            NEIGHBORSsort=NEIGHBORSsort[selection]      
    
        photoz[i]=np.inner(X[0],Ve[i])
        
        #photoz bound
        if bound==True:
            if photoz[i]< minz or photoz[i]>maxz:
               photozerr[i]+=np.abs(photoz[i]-NEIGHBORSsort[0]['z_true'])
               photoz[i]=NEIGHBORSsort[0]['z_true']
        
        percent=np.double(100*i)/Nvalid
        
        if i % 1000 ==1:
            print('progress: ',percent,'%')

    
    return photoz,photozerr,Vpdf,z1,nneighbors,closestNeighborDistance


