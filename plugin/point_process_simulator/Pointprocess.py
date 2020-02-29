import numpy as np
from matplotlib import pyplot, patches
import matplotlib.colors as colors
import scipy.stats as stats
import scipy.optimize as opt
import scipy.interpolate as intpol
from shapely.geometry import *
from shapely.ops import unary_union,triangulate
#from sklearn.model_selection import GridSearchCV
#from sklearn.neighbors import KernelDensity
import copy as copy

class Pointprocess(object):

    #S: list of multipoints, region: polygon, intFunc: callable
    def __init__(self,S=[],region=None,intFunc=lambda x,y:1):

        self.S=S
        self.region=region

        if(region):
            self.minBox=region.bounds
            f=lambda x,y: intFunc(x,y) if self.region.contains(Point(x,y)) else -0.0001
            self.intFunc=np.vectorize(f)
            if(S):
                nOfSamples=len(S)
                for n in range(nOfSamples):
                    S[n]=S[n].intersection(region)
        elif(S):
            self.minBox=unary_union(S).bounds
            self.region=unary_union(S).bounds
            self.intFunc=np.vectorize(intFunc)
        else:
            self.minBox=()
            self.intFunc=np.vectorize(intFunc)

    def plotProcess(self,func=None,k=-1):

        fig, ax=pyplot.subplots()
        xmin=self.minBox[0]
        ymin=self.minBox[1]
        xmax=self.minBox[2]
        ymax=self.minBox[3]

        if(self.region):
            patch=patches.Polygon(np.array(self.region.exterior),linestyle='-',linewidth=1,fill=False)
            ax.add_patch(patch)
        if(self.S):
            ax.scatter([p.x for p in self.S[k]],[p.y for p in self.S[k]],c='black',s=2)

            if(func):
                X,Y=np.meshgrid(np.linspace(xmin,xmax,200),np.linspace(ymin,ymax,200))
                func=np.vectorize(func)
                funcVals=func(X,Y)
                colorMap=copy.copy(pyplot.cm.coolwarm)
                colorMap.set_under('w')
                im=ax.imshow(funcVals, interpolation='bilinear',
                    cmap=colorMap,
                    norm=colors.Normalize(vmin=0),
                    aspect='auto',
                    origin='lower',
                    extent=[xmin,xmax,ymin,ymax],
                    alpha=0.9)
                ax.set_title('')
                cbar=fig.colorbar(im, extend='both', shrink=1, ax=ax)
                cbar.set_label('intensity')

            pyplot.plot()
            pyplot.show()

    def kernelIntensity(self, setIntensity=False):

        if(not self.S):
            print('no points')
            return

        data=np.array([[p.x,p.y] for p in self.S[-1]])
        N=len(self.S[-1])

        kde=stats.gaussian_kde(data.T)
        f=lambda x,y: N*kde.pdf(np.array([x,y])) if self.region.contains(Point(x,y)) else -0.0001

        kie=np.vectorize(f)

        if(setIntensity):
            self.intFunc=kie

        return kie


class PoissonProcess(Pointprocess):

    def __init__(self,S=[],region=None,intFunc=lambda x,y:1):
        Pointprocess.__init__(self,S,region,intFunc)

    def simHomogeneousPPP(self,intensity):

        N=np.random.poisson(lam=intensity*self.region.area)
        count=0
        accPoints=[]

        while(count<N):

            p=Point(np.random.uniform(self.minBox[0],self.minBox[2]),
                    np.random.uniform(self.minBox[1],self.minBox[3]))
            if(self.region.contains(p)):
                accPoints.append(p)
                count+=1

        return MultiPoint(accPoints)

    def simPPP(self,save=False,upper=None):

        if(not upper):
            out=opt.brute(lambda x:-1*self.intFunc(x[0],x[1]),
                                        ranges=((self.minBox[0],self.minBox[2],0.5),
                                        (self.minBox[1],self.minBox[3],0.5)),
                                        Ns=100,
                                        full_output=True)
            upper=-2*out[1]

        homPPP=self.simHomogeneousPPP(upper)
        self.S.append(homPPP)
        thinPP=[p for p in homPPP if upper*np.random.ranf()<=self.intFunc(p.x,p.y)]
        inHomPPP=MultiPoint(thinPP)

        if(save):
            self.S.append(inHomPPP)

        return inHomPPP

class LogGaussProcess(Pointprocess):

    def __init__(self,S=[],region=None,intFunc=lambda x,y:1,mean=lambda x,y:0):
        Pointprocess.__init__(self,S,region,intFunc)
        self.mean=mean
        self.randIntFunc=None

    def simGaussField(self,beta,sigma,N=300,corr='expon'):

        if(not self.region):
            print('Region required.')
            return

        PP=PoissonProcess(region=self.region)
        gridPoints=PP.simHomogeneousPPP(N/self.region.area).union(MultiPoint(self.region.exterior.coords))
        distMat=np.array([[p1.distance(p2) for p2 in gridPoints] for p1 in gridPoints])

        if(corr=='expon'):
            cov=sigma**2*np.exp(-1.*(1./beta*distMat)**2)
        elif(corr=='sine'):
            cardSine=np.vectorize(lambda x: np.sin(x)/(x) if x!=0 else 1)
            cov=sigma**2*cardSine(1./beta*distMat)
        elif(corr=='periodic'):
            cov=sigma**2*np.exp(2*np.sin(np.pi/0.2*distMat)**2/beta**2)
        else:
            pass

        meanVec=np.array([self.mean(p.x,p.y) for p in gridPoints])
        gaussField=np.random.multivariate_normal(meanVec,cov)
        upper=2*np.exp(np.max(gaussField))
        gridCoords=np.array([[p.x,p.y] for p in gridPoints])
        func=intpol.LinearNDInterpolator(gridCoords,gaussField)
        self.randIntFunc=lambda x,y: np.exp(func(x,y))

        return gridPoints, upper

    def simLGP(self,beta,sigma,save=False,N=300,corr='expon'):

        grid,upper=self.simGaussField(beta,sigma,N=N)
        PPP=PoissonProcess(region=self.region,intFunc=lambda x,y: self.randIntFunc(x,y))
        sample=PPP.simPPP(save=True,upper=upper)

        if(save):
            self.S.append(sample)

        return sample





if __name__=='__main__':

    #PP=Pointprocess(S=[MultiPoint([(0,0),(0,1),(1,1),(5,5)])],region=Polygon([(-1,-1),(2,-1),(2,2),(-1,3)]),intFunc=lambda x,y:(x-1)**2+(y-1)**2)
    #PP.plotProcess()
    PPP=PoissonProcess(region=Polygon([(0,0),(5,0),(5,5),(0,5)]),intFunc=lambda x,y:(5*np.sin((x-2.5)**2+(y-3)**2))**2)
    P=PPP.simPPP(save=True)
    PPP.plotProcess(PPP.intFunc)
    PPP.kernelIntensity(setIntensity=True)
    PPP.plotProcess(PPP.intFunc)


    #LGP=LogGaussProcess(region=Polygon([(0,0),(10,0),(10,10),(0,10)]))
    #d=LGP.simLGP(beta=2.5,sigma=1,save=True,corr='sine')
    #LGP.plotProcess(LGP.randIntFunc)










