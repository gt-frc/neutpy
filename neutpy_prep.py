#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Max Hill
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._cntr as cntr
from scipy.interpolate import griddata, UnivariateSpline
from scipy.constants import elementary_charge
from shapely.geometry import Point, LineString, LinearRing
from shapely.ops import polygonize,linemerge
from math import degrees, sqrt, acos, pi
from neutpy import neutpy,neutpyplot
import sys
import os
import re
from subprocess import call
import time

class neutpy_prep():
    def __init__(self, infile):
        self.read_infile(infile)
        if self.verbose: print 'sep_lines' 
        self.sep_lines()
        
        if self.verbose: print 'core_lines' 
        self.core_lines()
        
        if self.verbose: print 'sol_lines' 
        self.sol_lines()
        
        if self.verbose: print 'pfr_lines' 
        self.pfr_lines()
        
        if self.verbose: print 'core_nT' 
        self.core_nT()
        
        if self.verbose: print 'sol_nT' 
        self.sol_nT()
        
        if self.verbose: print 'pfr_nT' 
        self.pfr_nT()
        
        self.triangle_prep()
        self.read_triangle()

    @staticmethod
    def cut(line, distance):
        """Cuts a shapely line in two at a distance(normalized) from its starting point"""
        if distance <= 0.0 or distance >= 1.0:
            return [LineString(line)]
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p),normalized=True)
            if pd == distance:
                return [LineString(coords[:i+1]),LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance,normalized=True)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:])]
                
    @staticmethod
    def isinline(pt,line):
        pt_s = Point(pt)
        dist = line.distance(pt_s)
        if dist < 1E-6:
            return True
        else:
            return False

    @staticmethod
    def getangle(p1,p2):
        if isinstance(p1,Point) and isinstance(p2,Point):
            p1 = [p1.coords.xy[0][0],p1.coords.xy[1][0]]
            p2 = [p2.coords.xy[0][0],p2.coords.xy[1][0]]
        p1 = np.asarray(p1)
        p1 = np.reshape(p1,(-1,2))
        p2 = np.asarray(p2)
        p2 = np.reshape(p2,(-1,2))
        theta = np.arctan2(p1[:,1]-p2[:,1],p1[:,0]-p2[:,0])
        theta_mod = np.where(theta<0,theta+pi,theta) #makes it so the angle is always measured counterclockwise from the horizontal
        return theta
    
    @staticmethod
    def getangle3ptsdeg(p1,p2,p3):
        a = sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        b = sqrt((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)
        c = sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2) 
        theta = degrees(acos((c**2 - a**2 - b**2)/(-2*a*b))) #returns degree in radians
        return theta   

    @staticmethod
    def draw_line(R,Z,array,val,pathnum):
        res = cntr.Cntr(R,Z,array).trace(val)[pathnum]
        x = res[:,0]
        y = res[:,1]
        return x,y
        
    def read_infile(self,infile):
        #some regex commands we'll use when reading stuff in from the input file
        r0di = "r'%s *= *([ ,\d]*) *'%(v)"
        r0df = "r'%s *= *([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) *'%(v)"
        #r0ds = "r'%s *= *((?:/?\.?\w+\.?)+/?) *'%(v)"
        r0ds = "r'%s *= *((?:\/?\w+)+(?:\.\w+)?) *'%(v)"
        r1di = "r'%s\( *(\d*) *\) *= *(\d*) *'%(v)"
        r1df = "r'%s\( *(\d*)\) *= *([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) *'%(v)"
        r2df = "r'%s\( *(\d*)\) *= *((?:[+\-]?\d*\.?\d*(?:[eE]?[+\-]?\d+)?,?)*) *'%(v)"

        self.invars = {}
        self.invars["corelines_begin"]  = ["float", r0df]
        self.invars["num_corelines"]    = ["int",   r0di]
        self.invars["sollines_psi_max"] = ["float", r0df]
        self.invars["num_sollines"]     = ["int",   r0di]
        self.invars["xi_sep_pts"]       = ["int",   r0di]
        self.invars["ib_trim_off"]      = ["float", r0df]
        self.invars["ob_trim_off"]      = ["float", r0df]
        self.invars["xi_ib_pts"]        = ["int",   r0di]
        self.invars["xi_ob_pts"]        = ["int",   r0di]
        self.invars["BT0"]              = ["float", r0df]
        self.invars["verbose"]          = ["int",   r0di]
        self.invars["core_pol_pts"]     = ["int",   r0di]
        self.invars["ib_div_pol_pts"]   = ["int",   r0di]
        self.invars["ob_div_pol_pts"]   = ["int",   r0di]
        self.invars["pfr_ni_val"]       = ["float", r0df]
        self.invars["pfr_ne_val"]       = ["float", r0df]
        self.invars["pfr_Ti_val"]       = ["float", r0df]
        self.invars["pfr_Te_val"]       = ["float", r0df]
        
        self.in_prof = {}
        self.in_prof["ne_file"]         = ["str",   r0ds,   'ne_data']
        self.in_prof["ni_file"]         = ["str",   r0ds,   'ni_data']
        self.in_prof["Te_file"]         = ["str",   r0ds,   'Te_data']
        self.in_prof["Ti_file"]         = ["str",   r0ds,   'Ti_data']
        
        self.in_map2d = {}
        self.in_map2d["psirz_file"]     = ["str",   r0ds,   'psirz_exp']
        
        self.in_line2d = {}
        self.in_line2d["wall_file"]     = ["str",   r0ds,   'wall_exp']

        #Read input variables
        with open(os.getcwd() + '/inputs/' + infile, 'r') as f:
            for count, line in enumerate(f):
                if not line.startswith("#"):
                    #read in 0d variables
                    for v in self.invars:
                        exec("result = re.match(%s,line)"%(self.invars[v][1]))
                        if result:
                            exec("self.%s = %s(result.group(1))"%(v,self.invars[v][0]))
                            
                    #read in the names of radial profile input files 
                    for v in self.in_prof:
                        exec("result = re.match(%s,line)"%(self.in_prof[v][1]))
                        if result:
                            exec("self.%s = %s(result.group(1))"%(v,self.in_prof[v][0]))
   
                    #read in the names of input files that map a quantity on the R-Z plane
                    for v in self.in_map2d:
                        exec("result = re.match(%s,line)"%(self.in_map2d[v][1]))
                        if result:
                            exec("self.%s = %s(result.group(1))"%(v,self.in_map2d[v][0]))

                    #read in the names of input files that define a line in the R-Z plane 
                    for v in self.in_line2d:
                        #print 'v = ',count,v
                        exec("result = re.match(%s,line)"%(self.in_line2d[v][1]))
                        #print result
                        if result:
                            exec("self.%s = %s(result.group(1))"%(v,self.in_line2d[v][0])) 

        #read in additional input files 
        for infile in self.in_prof:
            try:
                exec("filename = self.%s"%(infile))
                filepath = os.getcwd()+'/inputs/'+ filename
                exec("self.%s = np.loadtxt('%s')"%(self.in_prof[infile][2],filepath))
            except:
                pass
            
        for infile in self.in_map2d:
            try:

                exec("filename = self.%s"%(infile))
                filepath = os.getcwd()+'/inputs/'+ filename
                exec("self.%s = np.loadtxt('%s')"%(self.in_map2d[infile][2],filepath))
            except:
                pass
            
        for infile in self.in_line2d:
            try:
                exec("filename = self.%s"%(infile))
                filepath = os.getcwd()+'/inputs/'+ filename
                exec("self.%s = np.loadtxt('%s')"%(self.in_line2d[infile][2],filepath))        
            except:
                pass

        self.wall_line = LineString(self.wall_exp)
        self.R         = self.psirz_exp[:,0].reshape(-1,65)
        self.Z         = self.psirz_exp[:,1].reshape(-1,65)
        self.psi       = self.psirz_exp[:,2].reshape(-1,65)
        
    def sep_lines(self):
        #find x-point location  
        dpsidR = np.gradient(self.psi,self.R[0,:],axis=1)
        dpsidZ = np.gradient(self.psi,self.Z[:,0],axis=0)
        d2psidR2 = np.gradient(dpsidR,self.R[0,:],axis=1)
        d2psidZ2 = np.gradient(dpsidZ,self.Z[:,0],axis=0)
        
        #find line(s) where dpsidR=0
        self.dpsidR_0 = cntr.Cntr(self.R,self.Z,dpsidR).trace(0.0)
        #find line(s) where dpsidZ=0
        self.dpsidZ_0 = cntr.Cntr(self.R,self.Z,dpsidZ).trace(0.0)
    
        for i,path1 in enumerate(self.dpsidR_0):
            for j,path2 in enumerate(self.dpsidZ_0):
                try:
                    #find intersection points between curves for dpsidR=0 and dpsidZ=0
                    ints = LineString(path1).intersection(LineString(path2))
                    #if there is only one intersection ('Point'), then we're probably not
                    #dealing with irrelevant noise in psi
                    if ints.type=='Point':
                        #check if local maximum or minimum
                        d2psidR2_pt = griddata(np.column_stack((self.R.flatten(),self.Z.flatten())),
                                 d2psidR2.flatten(),
                                 [ints.x,ints.y],
                                 method='cubic')
                        d2psidZ2_pt = griddata(np.column_stack((self.R.flatten(),self.Z.flatten())),
                                 d2psidZ2.flatten(),
                                 [ints.x,ints.y],
                                 method='cubic')
                        
                        if d2psidR2_pt>0 and d2psidZ2_pt>0:
                            #we've found the magnetic axis
                            self.m_axis = np.array([ints.x,ints.y])
                        elif d2psidR2_pt<0 and d2psidZ2_pt<0:
                            #we've found a magnet. Do nothing.
                            pass
                        elif ints.y<0:
                            #we've probably found our x-point, although this isn't super robust
                            #and obviously only applies to a single-diverted, lower-null configuration
                            #TODO: make this more robust, I could easily see this failing on some shots
                            self.xpt = np.array([ints.x,ints.y])
                            
                        #uncomment this line when debugging
                        #print list(ints.coords),d2psidR2(ints.x,ints.y),d2psidZ2(ints.x,ints.y)
                except:
                    pass
        
        #normalize psi
        psi_shift = self.psi + abs(np.amin(self.psi)) #set center to zero
        psi_shift_xpt = griddata(np.column_stack((self.R.flatten(),self.Z.flatten())),
                                 psi_shift.flatten(),
                                 self.xpt,
                                 method='cubic')
        #psi_shift_xpt = interp2d(R,Z,psi_shift,kind='linear')(xpt[0],xpt[1]) #get new value at sep
        self.psi_norm = psi_shift / psi_shift_xpt
        
        #create lines for seperatrix and divertor legs of seperatrix
        num_lines = int(len(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(1.0))/2)
        if num_lines==1:
            #in this case, the contour points that matplotlib returned constitute
            #a single line from inboard divertor to outboard divertor. We need to
            #add in the x-point in at the appropriate locations and split into a
            #main and a lower seperatrix line, each of which will include the x-point.
            x_psi,y_psi = self.draw_line(self.R,self.Z,self.psi_norm,1.0,0)
            
            
            loc1 = np.argmax(y_psi>self.xpt[1])
            loc2 = len(y_psi) - np.argmin(y_psi[::-1]<self.xpt[1])
    
            x_psi = np.insert(x_psi, (loc1,loc2), self.xpt[0])
            y_psi = np.insert(y_psi, (loc1,loc2), self.xpt[1])
            
            psi_1_pts = np.column_stack((x_psi,y_psi))
            self.main_sep_pts = psi_1_pts[loc1:loc2+1,:]
            self.main_sep_line = LineString(self.main_sep_pts[:-1])
            self.main_sep_line_closed = LineString(self.main_sep_pts)

            #get the inboard and outboard divertor legs seperately. This is so that
            #everything that includes the x-point can start with the x-point, which
            #elliminates the risk of tiny triangles in the vicinity of the x-point
            self.inboard_div_sep = np.flipud(psi_1_pts[:loc1+1])
            self.outboard_div_sep = psi_1_pts[loc2+1:]

            #cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.inboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ib_div_line = line
            self.ib_div_line_cut = self.cut(line,line.project(int_pt,normalized=True))[0]
            #self.ib_div_line_cut = line
            #TODO: add point to wall line
            
            
            #cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.outboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ob_div_line = line
            self.ob_div_line_cut = self.cut(line,line.project(int_pt,normalized=True))[0]
            
            ib_div_pts = np.flipud(np.asarray(self.ib_div_line_cut.xy).T)
            sep_pts    = np.asarray(self.main_sep_line.xy).T
            ob_div_pts = np.asarray(self.ob_div_line_cut.xy).T
            
            entire_sep_pts = np.vstack((ib_div_pts,sep_pts[1:,:],ob_div_pts))
            self.entire_sep_line = LineString(entire_sep_pts)

            #TODO: add point to wall line
                
        elif num_lines==2:
            #in this case, we have a lower seperatrix trace (line 0), and a main
            #seperatrix trace (line 1).
            
            #first do lower seperatrix line
            x_psi,y_psi = self.draw_line(self.R,self.Z,self.psi_norm,1.0,0)
            loc = np.argmax(x_psi>self.xpt[0])
            
            x_psi = np.insert(x_psi, loc, self.xpt[0])
            y_psi = np.insert(y_psi, loc, self.xpt[1])
            psi_1_pts = np.column_stack((x_psi,y_psi))
            
            self.inboard_div_sep = np.flipud(psi_1_pts[:loc+1])
            self.outboard_div_sep = psi_1_pts[loc+1:]
            
            #cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.inboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ib_div_line = line
            self.ib_div_line_cut = self.cut(line,line.project(int_pt,normalized=True))[0]

            #cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.outboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ob_div_line = line
            self.ob_div_line_cut = self.cut(line,line.project(int_pt,normalized=True))[0]
            #TODO: add point to wall line
    
            #now to main seperatrix line
            x_psi,y_psi = self.draw_line(self.R,self.Z,self.psi_norm,1.0,1)
            self.main_sep_pts = np.insert(np.column_stack((x_psi,y_psi)),0,self.xpt,axis=0)
            self.main_sep_line = LineString(self.main_sep_pts[:-1])
            self.main_sep_line_closed = LineString(self.main_sep_pts)
            
            entire_sep_pts = np.vstack((ib_div_pts,sep_pts[1:,:],ob_div_pts))
            self.entire_sep_line = LineString(entire_sep_pts)
            #now clean up the lines by removing any points that are extremely close
            #to the x-point 
            #TODO: 

    def core_lines(self):
        self.core_lines = []
        #psi_pts = np.concatenate((np.linspace(0,0.8,5,endpoint=False),np.linspace(0.8,1.0,4,endpoint=False)))
        psi_pts = np.linspace(self.corelines_begin,1,self.num_corelines,endpoint=False)
        for i,v in enumerate(psi_pts):
            num_lines = int(len(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(v))/2)
            if num_lines==1:
                #then we're definitely dealing with a surface inside the seperatrix
                x,y = self.draw_line(self.R,self.Z,self.psi_norm,v,0)
                self.core_lines.append(LineString(np.column_stack((x[:-1],y[:-1]))))
            else:
                #we need to find which of the surfaces is inside the seperatrix
                for j,line in enumerate(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(v)[:num_lines]):
                #for j,line in enumerate(cntr.Cntr(R,Z,self.psi_norm).trace(v)):
                    x,y = self.draw_line(self.R,self.Z,self.psi_norm,v,j)
                    if (np.amax(x) < np.amax(self.main_sep_pts[:,0]) and \
                        np.amin(x) > np.amin(self.main_sep_pts[:,0]) and \
                        np.amax(y) < np.amax(self.main_sep_pts[:,1]) and \
                        np.amin(y) > np.amin(self.main_sep_pts[:,1])):
                        #then it's an internal flux surface
                        self.core_lines.append(LineString(np.column_stack((x[:-1],y[:-1]))))
                        break
                    
    def sol_lines(self):
        #find value of psi at outside of what we're going to call the SOL
        self.sol_lines = []
        self.sol_lines_cut = []

        sol_width_obmp = 0.02
        psi_pts = np.linspace(1,self.sollines_psi_max,self.num_sollines+1,endpoint=True)[1:]
        for i,v in enumerate(psi_pts):
            num_lines = int(len(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(v))/2)
            if num_lines==1:
                #then we're definitely dealing with a surface inside the seperatrix
                x,y = self.draw_line(self.R,self.Z,self.psi_norm,v,0)
                self.sol_lines.append(LineString(np.column_stack((x,y))))
            else:
                #TODO: 
                pass
        for line in self.sol_lines:
            #find intersection points with the wall
            int_pts = line.intersection(self.wall_line)
            #cut line at intersection points
            cut_line = self.cut(line,line.project(int_pts[0],normalized=True))[1]
            cut_line = self.cut(cut_line,cut_line.project(int_pts[1],normalized=True))[0]
            self.sol_lines_cut.append(cut_line)
            
        #add wall intersection points from divertor legs and sol lines to wall_line.
        #This is necessary to prevent thousands of tiny triangles from forming if the 
        #end of the flux line isn't exactly on top of the wall line.

        #add inboard seperatrix strike point
        union = self.wall_line.union(self.ib_div_line)
        result = [geom for geom in polygonize(union)][0]
        self.wall_line = LineString(result.exterior.coords)

        #add outboard seperatrix strike point
        union = self.wall_line.union(self.ob_div_line)
        result = [geom for geom in polygonize(union)][0]
        self.wall_line = LineString(result.exterior.coords)  
        
        #add sol line intersection points on inboard side
        #for some reason, union freaks out when I try to do inboard and outboard
        #at the same time.
        for num,line in enumerate(self.sol_lines):
            union = self.wall_line.union(self.cut(line,0.5)[0])    
            result = [geom for geom in polygonize(union)][0]
            self.wall_line = LineString(result.exterior.coords)

        #add sol line intersection points on outboard side            
        for num,line in enumerate(self.sol_lines):
            union = self.wall_line.union(self.cut(line,0.5)[1])    
            result = [geom for geom in polygonize(union)][0]
            self.wall_line = LineString(result.exterior.coords)
        
    def pfr_lines(self):
        num_lines = int(len(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(0.999))/2)
        if num_lines==1:
            #then we're definitely dealing with a surface inside the seperatrix
            print 'Did not find PFR flux surface. Stopping.'
            sys.exit()
        else:
            #we need to find the surface that is contained within the private flux region
            for j,line in enumerate(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(0.99)[:num_lines]):
            #for j,line in enumerate(cntr.Cntr(R,Z,self.psi_norm).trace(v)):
                x,y = self.draw_line(self.R,self.Z,self.psi_norm,0.99,j)
                if (np.amax(y) < np.amin(self.main_sep_pts[:,1])):
                    #then it's a pfr flux surface, might need to add additional checks later
                    pfr_line_raw = LineString(np.column_stack((x,y)))
                    #find cut points
                    cut_pt1 = pfr_line_raw.intersection(self.wall_line)[0]
                    dist1   = pfr_line_raw.project(cut_pt1,normalized=True)
                    cutline_temp  = self.cut(pfr_line_raw,dist1)[1]
                    
                    #reverse line point order so we can reliably find the second intersection point
                    cutline_temp_rev = LineString(np.flipud(np.asarray(cutline_temp.xy).T))
                    
                    cut_pt2 = cutline_temp_rev.intersection(self.wall_line)
                    dist2   = cutline_temp_rev.project(cut_pt2,normalized=True)
                    cutline_final_rev  = self.cut(cutline_temp_rev,dist2)[1] 

                    #reverse again for final pfr flux line
                    pfr_flux_line = LineString(np.flipud(np.asarray(cutline_final_rev.xy).T))
                    
                    #add pfr_line intersection points on inboard side
                    #for some reason, union freaks out when I try to do inboard and outboard
                    #at the same time.
                    union = self.wall_line.union(self.cut(pfr_line_raw,0.5)[0])    
                    result = [geom for geom in polygonize(union)][0]
                    self.wall_line = LineString(result.exterior.coords)
            
                    #add pfr line intersection points on outboard side   
                    union = self.wall_line.union(self.cut(pfr_line_raw,0.5)[1])    
                    result = [geom for geom in polygonize(union)][0]
                    self.wall_line = LineString(result.exterior.coords)

                    #cut out pfr section of wall line
                    wall_pts = np.asarray(self.wall_line.xy).T

                    #ib_int_pt = np.asarray(self.ib_div_line.intersection(self.wall_line).xy).T
                    #ob_int_pt = self.ob_div_line.intersection(self.wall_line)
                    wall_start_pos = np.where((wall_pts==cut_pt2).all(axis=1))[0][0]
                    wall_line_rolled = LineString(np.roll(wall_pts,-wall_start_pos,axis=0))
                    wall_line_cut_pfr = self.cut(wall_line_rolled, 
                                             wall_line_rolled.project(cut_pt1,normalized=True))[0]
                    
                    #create LineString with pfr line and section of wall line
                    self.pfr_line = linemerge((pfr_flux_line,wall_line_cut_pfr))
                    break
        #plt.axis('equal')
        #plt.plot(np.asarray(self.wall_line.xy).T[:,0],np.asarray(self.wall_line.xy).T[:,1],color='black',lw=0.5)
        #plt.plot(np.asarray(self.pfr_line.xy).T[:,0],np.asarray(self.pfr_line.xy).T[:,1],color='red',lw=0.5)
        

    def core_nT(self):
        
        #Master arrays that will contain all the points we'll use to get n,T
        #throughout the plasma chamber via 2-D interpolation
        self.ni_pts = np.zeros((0,3),dtype='float')
        self.ne_pts = np.zeros((0,3),dtype='float')
        self.Ti_kev_pts = np.zeros((0,3),dtype='float')
        self.Te_kev_pts = np.zeros((0,3),dtype='float')
        
        ##########################################
        #Calculate n,T throughout the core plasma using radial profile input files, uniform on flux surface
        
        ni     = UnivariateSpline(self.ni_data[:,0],self.ni_data[:,1],k=5,s=2.0)
        ne     = UnivariateSpline(self.ne_data[:,0],self.ne_data[:,1],k=5,s=2.0)
        Ti_kev = UnivariateSpline(self.Ti_data[:,0],self.Ti_data[:,1],k=5,s=2.0)
        Te_kev = UnivariateSpline(self.Te_data[:,0],self.Te_data[:,1],k=5,s=2.0)
        
        #get approximate rho values associated with the psi values we're using
        #draw line between magnetic axis and the seperatrix at the outboard midplane
        self.obmp_pt = self.main_sep_pts[np.argmax(self.main_sep_pts,axis=0)[0]]
        self.ibmp_pt = self.main_sep_pts[np.argmin(self.main_sep_pts,axis=0)[0]]
        self.top_pt  = self.main_sep_pts[np.argmax(self.main_sep_pts,axis=0)[1]]
        self.bot_pt  = self.main_sep_pts[np.argmin(self.main_sep_pts,axis=0)[1]]

        rho_line = LineString([Point(self.m_axis),Point(self.obmp_pt)])
        #for several points on the rho line specified above:
        
        #To get smooth gradients for use in the SOL calculation, you need around
        # 50-100 radial points in the far edge and around 100 or so theta points
        # TODO: There is almost certainly a faster way to get these gradients.
        rho_pts = np.concatenate((np.linspace(0, 0.95, 20, endpoint=False), 
                                 np.linspace(0.95, 1, 50, endpoint=False)),axis=0)
        
        thetapts = np.linspace(0,1,100,endpoint=False)
        for i,rho in enumerate(rho_pts): 
            #get n,T information at the point by interpolating the rho-based input file data
            ni_val = ni(rho)
            ne_val = ne(rho)
            Ti_kev_val = Ti_kev(rho)
            Te_kev_val = Te_kev(rho)
            #get R,Z coordinates of each point along the rho_line
            pt_coords = np.asarray(rho_line.interpolate(rho,normalized=True).coords)[0]

            #get psi value at that point
            psi_val = griddata(np.column_stack((self.R.flatten(),self.Z.flatten())),
                                 self.psi_norm.flatten(),
                                 pt_coords,
                                 method='linear')
            #map this n,T data to every point on the corresponding flux surface
            num_lines = int(len(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(psi_val))/2)

            if num_lines==1:
                #then we're definitely dealing with a surface inside the seperatrix
                x,y = self.draw_line(self.R,self.Z,self.psi_norm,psi_val,0)
                surf = LineString(np.column_stack((x,y)))
            else:
                #we need to find which of the surfaces is inside the seperatrix
                for j,line in enumerate(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(psi_val)[:num_lines]):
                    #for j,line in enumerate(cntr.Cntr(R,Z,self.psi_norm).trace(v)):
                    x,y = self.draw_line(self.R,self.Z,self.psi_norm,psi_val,j)
                    if (np.amax(x) < np.amax(self.main_sep_pts[:,0]) and \
                        np.amin(x) > np.amin(self.main_sep_pts[:,0]) and \
                        np.amax(y) < np.amax(self.main_sep_pts[:,1]) and \
                        np.amin(y) > np.amin(self.main_sep_pts[:,1])):
                        #then it's an internal flux surface
                        surf = LineString(np.column_stack((x,y)))
                        break
            
            for j,theta_norm in enumerate(thetapts):
                pt = np.asarray(surf.interpolate(theta_norm,normalized=True).coords).T
                self.ni_pts = np.vstack((self.ni_pts,np.append(pt,ni_val)))
                self.ne_pts = np.vstack((self.ne_pts,np.append(pt,ne_val)))
                self.Ti_kev_pts = np.vstack((self.Ti_kev_pts,np.append(pt,Ti_kev_val)))
                self.Te_kev_pts = np.vstack((self.Te_kev_pts,np.append(pt,Te_kev_val)))

        #Do seperatrix separately so we don't accidentally assign the input n,T data to the divertor legs
        self.ni_sep_val = ni(1.0)
        self.ne_sep_val = ne(1.0)
        self.Ti_kev_sep_val = Ti_kev(1.0)
        self.Te_kev_sep_val = Te_kev(1.0)
        self.Ti_J_sep_val = self.Ti_kev_sep_val * 1.0E3 * 1.6021E-19
        self.Te_J_sep_val = self.Te_kev_sep_val * 1.0E3 * 1.6021E-19
        for j,theta_norm in enumerate(thetapts): 
            pt = np.asarray(self.main_sep_line.interpolate(theta_norm,normalized=False).coords,dtype='float').T
            self.ni_pts = np.vstack((self.ni_pts,np.append(pt,self.ni_sep_val)))
            self.ne_pts = np.vstack((self.ne_pts,np.append(pt,self.ne_sep_val)))
            self.Ti_kev_pts = np.vstack((self.Ti_kev_pts,np.append(pt,self.Ti_kev_sep_val)))
            self.Te_kev_pts = np.vstack((self.Te_kev_pts,np.append(pt,self.Te_kev_sep_val)))



    def sol_nT(self):
        #Calculate n,T in SOL using Bohm diffusion, core data from radial profile input files, and input
        #divertor target densities and temperatures (replace with 2-pt divertor model later)

        #draw core line just inside the seperatrix (seperatrix would be too noisy absent SOL data, which is what we're trying to calculate)
        psi_val = 0.98
        num_lines = int(len(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(psi_val))/2)
        if num_lines==1:
            #then we're definitely dealing with a surface inside the seperatrix
            x,y = self.draw_line(self.R,self.Z,self.psi_norm,psi_val,0)
        else:
            #we need to find which of the surfaces is inside the seperatrix
            for j,line in enumerate(cntr.Cntr(self.R,self.Z,self.psi_norm).trace(psi_val)[:num_lines]):
            #for j,line in enumerate(cntr.Cntr(R,Z,self.psi_norm).trace(v)):
                x,y = self.draw_line(self.R,self.Z,self.psi_norm,psi_val,j)
                if (np.amax(x) < np.amax(self.main_sep_pts[:,0]) and \
                    np.amin(x) > np.amin(self.main_sep_pts[:,0]) and \
                    np.amax(y) < np.amax(self.main_sep_pts[:,1]) and \
                    np.amin(y) > np.amin(self.main_sep_pts[:,1])):
                    #then it's an internal flux surface
                    break
        
        #get quantities on a fairly fine R,Z grid for the purpose of taking gradients, etc.
        R_temp,Z_temp = np.meshgrid(np.linspace(0.95*self.ibmp_pt[0],1.05*self.obmp_pt[0],500),
                                    np.linspace(1.05*self.top_pt[1],1.05*self.bot_pt[1],500))

        ni_grid = griddata(self.ni_pts[:,:-1],
                           self.ni_pts[:,-1],
                           (R_temp,Z_temp),
                           method='cubic',
                           fill_value = self.ni_sep_val)
        ne_grid = griddata(self.ne_pts[:,:-1],
                           self.ne_pts[:,-1],
                           (R_temp,Z_temp),
                           method='cubic',
                           fill_value = self.ne_sep_val)
        Ti_grid = griddata(self.Ti_kev_pts[:,:-1],
                           self.Ti_kev_pts[:,-1]*1.0E3*1.6021E-19,
                           (R_temp,Z_temp),
                           method='cubic',
                           fill_value = self.Ti_J_sep_val)
        Te_grid = griddata(self.Te_kev_pts[:,:-1],
                           self.Te_kev_pts[:,-1]*1.0E3*1.6021E-19,
                           (R_temp,Z_temp),
                           method='cubic',
                           fill_value = self.Te_J_sep_val)
        
        dnidr = -1.0*(np.abs(np.gradient(ni_grid,Z_temp[:,0],axis=1)) + np.abs(np.gradient(ni_grid,R_temp[0,:],axis=0)))
        dnedr = -1.0*(np.abs(np.gradient(ne_grid,Z_temp[:,0],axis=1)) + np.abs(np.gradient(ne_grid,R_temp[0,:],axis=0)))
        dTidr = -1.0*(np.abs(np.gradient(Ti_grid,Z_temp[:,0],axis=1)) + np.abs(np.gradient(Ti_grid,R_temp[0,:],axis=0)))
        dTedr = -1.0*(np.abs(np.gradient(Te_grid,Z_temp[:,0],axis=1)) + np.abs(np.gradient(Te_grid,R_temp[0,:],axis=0)))
                
        #Get densities, temperatures, and other quantities along the flux surface we just drew
        #note densities and temperatures on seperatrix are assumed to be constant for all theta and are
        #obtained above, i.e. self.ni_sep_val, etc.
        dnidr_sep_raw = griddata(np.column_stack((R_temp.flatten(),Z_temp.flatten())),
                             dnidr.flatten(),
                             np.column_stack((x,y)),
                             method='cubic'
                             )
        dnedr_sep_raw = griddata(np.column_stack((R_temp.flatten(),Z_temp.flatten())),
                             dnedr.flatten(),
                             np.column_stack((x,y)),
                             method='cubic'
                             )
        
        dTidr_sep_raw = griddata(np.column_stack((R_temp.flatten(),Z_temp.flatten())),
                             dTidr.flatten(),
                             np.column_stack((x,y)),
                             method='cubic'
                             )
        
        dTedr_sep_raw = griddata(np.column_stack((R_temp.flatten(),Z_temp.flatten())),
                             dTedr.flatten(),
                             np.column_stack((x,y)),
                             method='cubic'
                             )
        
        BT_sep_raw     = griddata(np.column_stack((R_temp.flatten(),Z_temp.flatten())),
                             (self.m_axis[0]*self.BT0/R_temp).flatten(),
                             np.column_stack((x,y)),
                             method='cubic'
                             )
        
        #Get densities, temperatures, and other quantities along the inboard divertor leg of seperatrix
        #doing linear interpolation in the absense of a 1D model
        
        
        #Get densities, temperatures, and other quantities along the outboard divertor leg of seperatrix

        
        #norm factor used to divide by the order of magnitude to facilitate easier smoothing
        ni_norm_factor = 1.0#*10**(int(np.log10(np.average(dnidr_sep)))-1)
        ne_norm_factor = 1.0#*10**(int(np.log10(np.average(dnedr_sep)))-1)
        Ti_norm_factor = 1.0#*10**(int(np.log10(np.average(dTidr_sep)))-1)
        Te_norm_factor = 1.0#*10**(int(np.log10(np.average(dTedr_sep)))-1)
        
        #specify the number of xi (parallel) points in the seperatrix region of the 2 point divertor model

        
        ni_sep = np.zeros(self.xi_sep_pts) + self.ni_sep_val
        ne_sep = np.zeros(self.xi_sep_pts) + self.ne_sep_val
        Ti_sep = np.zeros(self.xi_sep_pts) + self.Ti_J_sep_val
        Te_sep = np.zeros(self.xi_sep_pts) + self.Te_J_sep_val
        
        dnidr_sep = UnivariateSpline(np.linspace(0,1,len(dnidr_sep_raw)),
                                            dnidr_sep_raw/ni_norm_factor,
                                            k=5,
                                            s=0.0)(np.linspace(self.ib_trim_off,1.0-self.ob_trim_off,self.xi_sep_pts))*ni_norm_factor

        dnedr_sep = UnivariateSpline(np.linspace(0,1,len(dnedr_sep_raw)),
                                            dnedr_sep_raw/ne_norm_factor,
                                            k=5,
                                            s=0.0)(np.linspace(self.ib_trim_off,1.0-self.ob_trim_off,self.xi_sep_pts))*ne_norm_factor

        dTidr_sep = UnivariateSpline(np.linspace(0,1,len(dTidr_sep_raw)),
                                            dTidr_sep_raw/Ti_norm_factor,
                                            k=5,
                                            s=0.0)(np.linspace(self.ib_trim_off,1.0-self.ob_trim_off,self.xi_sep_pts))*Ti_norm_factor
                                            
        dTedr_sep = UnivariateSpline(np.linspace(0,1,len(dTedr_sep_raw)),
                                            dTedr_sep_raw/Te_norm_factor,
                                            k=5,
                                            s=0.0)(np.linspace(self.ib_trim_off,1.0-self.ob_trim_off,self.xi_sep_pts))*Te_norm_factor
                                            
        BT_sep    = UnivariateSpline(np.linspace(0,1,len(dnidr_sep_raw)),
                                            BT_sep_raw,
                                            k=5,
                                            s=0.0)(np.linspace(self.ib_trim_off,1.0-self.ob_trim_off,self.xi_sep_pts))
        
        ni_ib_wall = self.ni_sep_val * 1.0
        ni_ob_wall = self.ni_sep_val * 1.0
        ni_ib = np.linspace(ni_ib_wall,self.ni_sep_val,self.xi_ib_pts,endpoint=False)
        ni_ob = np.linspace(self.ni_sep_val,ni_ob_wall,self.xi_ob_pts,endpoint=True)     
        
        ne_ib_wall = self.ne_sep_val * 1.0
        ne_ob_wall = self.ne_sep_val * 1.0
        ne_ib = np.linspace(ne_ib_wall,self.ne_sep_val,self.xi_ib_pts,endpoint=False)
        ne_ob = np.linspace(self.ne_sep_val,ne_ob_wall,self.xi_ob_pts,endpoint=True) 
        
        Ti_ib_wall = self.Ti_J_sep_val * 1.0
        Ti_ob_wall = self.Ti_J_sep_val * 1.0
        Ti_ib = np.linspace(Ti_ib_wall,self.Ti_J_sep_val,self.xi_ib_pts,endpoint=False)
        Ti_ob = np.linspace(self.Ti_J_sep_val,Ti_ob_wall,self.xi_ob_pts,endpoint=True)    

        Te_ib_wall = self.Te_J_sep_val * 1.0
        Te_ob_wall = self.Te_J_sep_val * 1.0
        Te_ib = np.linspace(Te_ib_wall,self.Te_J_sep_val,self.xi_ib_pts,endpoint=False)
        Te_ob = np.linspace(self.Te_J_sep_val,Te_ob_wall,self.xi_ob_pts,endpoint=True)    

        dnidr_ib_wall = dnidr_sep[0]
        dnidr_ob_wall = dnidr_sep[-1]
        dnidr_ib = np.linspace(dnidr_ib_wall,dnidr_sep[0],self.xi_ib_pts,endpoint=False)
        dnidr_ob = np.linspace(dnidr_sep[-1],dnidr_ob_wall,self.xi_ob_pts,endpoint=True)
        
        dnedr_ib_wall = dnedr_sep[0]
        dnedr_ob_wall = dnedr_sep[-1]
        dnedr_ib = np.linspace(dnedr_ib_wall,dnedr_sep[0],self.xi_ib_pts,endpoint=False)
        dnedr_ob = np.linspace(dnedr_sep[-1],dnedr_ob_wall,self.xi_ob_pts,endpoint=True)
        
        dTidr_ib_wall = dTidr_sep[0]
        dTidr_ob_wall = dTidr_sep[-1]
        dTidr_ib = np.linspace(dTidr_ib_wall,dTidr_sep[0],self.xi_ib_pts,endpoint=False)
        dTidr_ob = np.linspace(dTidr_sep[-1],dTidr_ob_wall,self.xi_ob_pts,endpoint=True)
        
        dTedr_ib_wall = dTedr_sep[0]
        dTedr_ob_wall = dTedr_sep[-1]
        dTedr_ib = np.linspace(dTedr_ib_wall,dTedr_sep[0],self.xi_ib_pts,endpoint=False)
        dTedr_ob = np.linspace(dTedr_sep[-1],dTedr_ob_wall,self.xi_ob_pts,endpoint=True)
        
        BT_ib_wall = BT_sep[0]
        BT_ob_wall = BT_sep[-1]
        BT_ib = np.linspace(BT_ib_wall,BT_sep[0],self.xi_ib_pts,endpoint=False)
        BT_ob = np.linspace(BT_sep[-1],BT_ob_wall,self.xi_ob_pts,endpoint=True) 
        
        ni_xi    = np.concatenate((ni_ib,ni_sep,ni_ob))
        ne_xi    = np.concatenate((ne_ib,ne_sep,ne_ob))
        Ti_xi    = np.concatenate((Ti_ib,Ti_sep,Ti_ob))
        Te_xi    = np.concatenate((Te_ib,Te_sep,Te_ob))
        dnidr_xi = np.concatenate((dnidr_ib,dnidr_sep,dnidr_ob))
        dnedr_xi = np.concatenate((dnedr_ib,dnedr_sep,dnedr_ob))
        dTidr_xi = np.concatenate((dTidr_ib,dTidr_sep,dTidr_ob))
        dTedr_xi = np.concatenate((dTedr_ib,dTedr_sep,dTedr_ob))
        BT_xi    = np.concatenate((BT_ib,BT_sep,BT_ob))
        
        ib_leg_length = self.ib_div_line_cut.length
        ob_leg_length = self.ob_div_line_cut.length
        sep_length    = self.main_sep_line_closed.length
        ib_frac  = ib_leg_length / (ib_leg_length + sep_length + ob_leg_length)
        sep_frac = sep_length    / (ib_leg_length + sep_length + ob_leg_length)
        ob_frac  = ob_leg_length / (ib_leg_length + sep_length + ob_leg_length)
  

        xi_ib_div = np.linspace(0,
                                ib_frac+sep_frac*self.ib_trim_off,
                                self.xi_ib_pts,
                                endpoint=False)
        
        xi_sep    = np.linspace(ib_frac+sep_frac*self.ib_trim_off,
                                ib_frac+sep_frac*self.ib_trim_off + sep_frac-(self.ib_trim_off + self.ob_trim_off),
                                self.xi_sep_pts,
                                endpoint=False)

        xi_ob_div    = np.linspace(ib_frac+sep_frac*self.ib_trim_off + sep_frac-(self.ib_trim_off + self.ob_trim_off),
                                1,
                                self.xi_ob_pts,
                                endpoint=True)
        
        xi_pts = np.concatenate((xi_ib_div,xi_sep,xi_ob_div))
                                        
        #model perpendicular particle and heat transport using Bohm Diffusion
        D_perp   =       Ti_xi / (16.0 * elementary_charge * BT_xi)
        Chi_perp = 5.0 * Ti_xi / (32.0 * elementary_charge * BT_xi)
        
        Gamma_perp = -D_perp * dnidr_xi
        Q_perp     = -ni_xi * Chi_perp * dTidr_xi - \
                     3.0 * Ti_xi * D_perp * dnidr_xi
        
        delta_sol_n = D_perp * ni_xi / Gamma_perp
        delta_sol_T = Chi_perp / \
                    (Q_perp/(ni_xi*Ti_xi) \
                     - 3.0*D_perp/delta_sol_n)
        delta_sol_E = 2/7*delta_sol_T
        #plt.plot(delta_sol_n)

        
        #plt.axis('equal')
        #plt.contourf(R_temp,
        #             Z_temp,
        #             ni_grid,
        #             500)
        #plt.colorbar()

        #plt.plot(xi_pts,ni_xi)
        #plt.plot(dnidr_sep_raw)
        #plt.plot(xi_pts,dTidr_sep_smooth)
        #plt.plot(xi_pts,np.nan_to_num(-Ti_J_sep_val/dTidr_sep_smooth))
        #plt.plot(xi_pts,BT_sep)
        #plt.plot(xi_pts,BT_sep_smooth)
        #plt.plot(xi_pts,D_perp,label='D_perp')
        #plt.plot(xi_pts,Chi_perp,label='Chi_perp')
        #plt.plot(xi_pts,Gamma_perp,label='Gamma_perp')
        #plt.plot(xi_pts,Q_perp,label='Q_perp')
        #plt.plot(xi_pts,-ni_sep_val * Chi_perp * dTidr_sep_smooth,label='Q term 1')
        #plt.plot(xi_pts,3.0 * Ti_J_sep_val * D_perp * dnidr_sep_smooth,label='Q term 2')
        #plt.plot(xi_pts,Q_perp,label='Q_perp')
        #plt.plot(xi_pts,3.0*D_perp*ni_sep_val*Ti_J_sep_val/delta_sol_n,label='term2')
        #plt.plot(xi_pts,delta_sol_n,label='delta_sol_n')
        #plt.plot(xi_pts,delta_sol_T,label='delta_sol_T')
        #plt.plot(xi_pts,delta_sol_E,label='delta_sol_E')
        #plt.legend()
        #plt.plot()
        #sys.exit()
        #delta_n_xi_ib  = np.array([0])
        #delta_n_xi_ib  = np.array([0])
        
        #pts = np.concatenate((np.array([0.0]),
        #                      np.linspace(ib_frac,ib_frac+sep_frac,xi_sep_pts),
        #                      np.array([1.0])))
        #vals = np.concatenate((delta_n_xi_ib,delta_sol_n_trim,delta_n_xi_ib))

        #delta_n_xi_sep = griddata(pts,
        #         vals,
        #         xi_pts,
        #         method='linear',
        #         fill_value='np.nan')


        r_max  = 0.45
        twoptdiv_r_pts = 20
        
        r_pts  = np.linspace(0,r_max,twoptdiv_r_pts)
        xi,r = np.meshgrid(xi_pts,r_pts)
        sol_ni =   ni_xi * np.exp(-r/delta_sol_n)
        sol_ne =   ne_xi * np.exp(-r/delta_sol_n)
        sol_Ti =   Ti_xi * np.exp(-r/delta_sol_T)
        sol_Te =   Te_xi * np.exp(-r/delta_sol_T)

        
        #draw sol lines through 2d strip model to get n,T along the lines
        sol_line_dist = np.zeros((len(xi_pts),len(self.sol_lines_cut)))
        sol_nT_pts = np.zeros((len(xi_pts),2,len(self.sol_lines_cut)))
        for i,sol_line in enumerate(self.sol_lines_cut):
            for j, xi_val in enumerate(xi_pts):
                sol_pt = sol_line.interpolate(xi_val,normalized=True)
                sol_nT_pts[j,:,i] = np.asarray(sol_pt.xy).T
                sep_pt_pos = self.entire_sep_line.project(sol_pt,normalized=True)
                sep_pt = self.entire_sep_line.interpolate(sep_pt_pos,normalized=True)
                sol_line_dist[j,i] = sol_pt.distance(sep_pt)
        
        sol_line_ni = np.zeros((len(xi_pts),len(self.sol_lines_cut)))
        sol_line_ne = np.zeros((len(xi_pts),len(self.sol_lines_cut)))
        sol_line_Ti = np.zeros((len(xi_pts),len(self.sol_lines_cut)))
        sol_line_Te = np.zeros((len(xi_pts),len(self.sol_lines_cut)))
        for i,sol_line in enumerate(self.sol_lines_cut):
            sol_line_ni[:,i] = griddata(np.column_stack((xi.flatten(),r.flatten())),
                                        sol_ni.flatten(),
                                        np.column_stack((np.linspace(0,1,len(xi_pts)),sol_line_dist[:,i])),
                                        method='linear')
            sol_line_ne[:,i] = griddata(np.column_stack((xi.flatten(),r.flatten())),
                                        sol_ne.flatten(),
                                        np.column_stack((np.linspace(0,1,len(xi_pts)),sol_line_dist[:,i])),
                                        method='linear')
            sol_line_Ti[:,i] = griddata(np.column_stack((xi.flatten(),r.flatten())),
                                        sol_Ti.flatten(),
                                        np.column_stack((np.linspace(0,1,len(xi_pts)),sol_line_dist[:,i])),
                                        method='linear')
            sol_line_Te[:,i] = griddata(np.column_stack((xi.flatten(),r.flatten())),
                                        sol_Te.flatten(),
                                        np.column_stack((np.linspace(0,1,len(xi_pts)),sol_line_dist[:,i])),
                                        method='linear')
        
        
        #append to master arrays
        for i,line in enumerate(self.sol_lines_cut):
            pts_ni_sol = np.column_stack((sol_nT_pts[:,:,i],sol_line_ni[:,i]))
            pts_ne_sol = np.column_stack((sol_nT_pts[:,:,i],sol_line_ne[:,i]))
            pts_Ti_sol = np.column_stack((sol_nT_pts[:,:,i],sol_line_Ti[:,i]/1.0E3/1.6021E-19)) #converting back to kev
            pts_Te_sol = np.column_stack((sol_nT_pts[:,:,i],sol_line_Te[:,i]/1.0E3/1.6021E-19)) #converting back to kev
            
            self.ni_pts     = np.vstack((self.ni_pts,pts_ni_sol))
            self.ne_pts     = np.vstack((self.ne_pts,pts_ne_sol))
            self.Ti_kev_pts = np.vstack((self.Ti_kev_pts,pts_Ti_sol))
            self.Te_kev_pts = np.vstack((self.Te_kev_pts,pts_Te_sol))

        #draw wall line through 2d strip model to get n,T along the line
        wall_pts = np.asarray(self.wall_line.xy).T
        ib_int_pt = np.asarray(self.ib_div_line.intersection(self.wall_line).xy).T
        ob_int_pt = self.ob_div_line.intersection(self.wall_line)
        wall_start_pos = np.where((wall_pts==ib_int_pt).all(axis=1))[0][0]
        wall_line_rolled = LineString(np.roll(wall_pts,-wall_start_pos,axis=0))
        wall_line_cut = self.cut(wall_line_rolled, 
                            wall_line_rolled.project(ob_int_pt,normalized=True))[0]
        #add points to wall line for the purpose of getting n,T along the wall. These points
        #won't be added to the main wall line or included in the triangulation.
        #for i,v in enumerate(np.linspace(0,1,300)):
        #    #interpolate along wall_line_cut to find point to add
        #    pt = wall_line_cut.interpolate(v,normalized=True)
        #    #add point to wall_line_cut
        #    union = wall_line_cut.union(pt)
        #    result = [geom for geom in polygonize(union)][0]
        #    wall_line_cut = LineString(result.exterior.coords)
        
        wall_nT_pts = np.asarray(wall_line_cut)
        num_wall_pts = len(wall_nT_pts)
        wall_pos_norm = np.zeros(num_wall_pts)
        wall_dist = np.zeros(num_wall_pts)

        for i,pt in enumerate(wall_nT_pts):
            wall_pt = Point(pt)
            sep_pt_pos = self.entire_sep_line.project(Point(wall_pt),normalized=True)
            sep_pt = self.entire_sep_line.interpolate(sep_pt_pos,normalized=True)
            wall_pos_norm[i] = wall_line_cut.project(wall_pt,normalized=True)
            wall_dist[i] = wall_pt.distance(sep_pt)
        
        wall_ni = griddata(np.column_stack((xi.flatten(),r.flatten())),
                           sol_ni.flatten(),
                           np.column_stack((wall_pos_norm,wall_dist)),
                           method='linear')
        wall_ne = griddata(np.column_stack((xi.flatten(),r.flatten())),
                           sol_ne.flatten(),
                           np.column_stack((wall_pos_norm,wall_dist)),
                           method='linear')
        wall_Ti = griddata(np.column_stack((xi.flatten(),r.flatten())),
                           sol_Ti.flatten(),
                           np.column_stack((wall_pos_norm,wall_dist)),
                           method='linear')
        wall_Te = griddata(np.column_stack((xi.flatten(),r.flatten())),
                           sol_Te.flatten(),
                           np.column_stack((wall_pos_norm,wall_dist)),
                           method='linear')
        
        #set minimum wall densities and temperatures
        #TODO: this needs to be more robust
        wall_ni_min = 1.0E15
        wall_ne_min = 1.0E15
        wall_Ti_min = 0.02 * 1.0E3 * 1.6021E-19
        wall_Te_min = 0.02 * 1.0E3 * 1.6021E-19
        
        wall_ni[wall_ni < wall_ni_min] = wall_ni_min
        wall_ne[wall_ne < wall_ne_min] = wall_ne_min
        wall_Ti[wall_Ti < wall_Ti_min] = wall_Ti_min
        wall_Te[wall_Te < wall_Te_min] = wall_Te_min
        
        
        #append to master arrays
        pts_ni_wall = np.column_stack((wall_nT_pts,wall_ni))
        pts_ne_wall = np.column_stack((wall_nT_pts,wall_ne))
        pts_Ti_wall = np.column_stack((wall_nT_pts,wall_Ti/1.0E3/1.6021E-19)) #converting back to kev
        pts_Te_wall = np.column_stack((wall_nT_pts,wall_Te/1.0E3/1.6021E-19)) #converting back to kev
        
        self.ni_pts     = np.vstack((self.ni_pts,pts_ni_wall))
        self.ne_pts     = np.vstack((self.ne_pts,pts_ne_wall))
        self.Ti_kev_pts = np.vstack((self.Ti_kev_pts,pts_Ti_wall))
        self.Te_kev_pts = np.vstack((self.Te_kev_pts,pts_Te_wall))
        
        #plt.contourf(xi,r,np.log10(sol_ni),500)
        #plt.colorbar()
        #for i,v in enumerate(self.sol_lines_cut):
        #    plt.plot(xi_pts,sol_line_dist[:,i])
        #plt.plot(np.linspace(0,1,num_wall_pts),wall_dist,color='black')

    def pfr_nT(self):
        
        pfr_pts = np.asarray(self.pfr_line.xy).T
        pfr_ni = np.zeros(len(pfr_pts)) + self.pfr_ni_val
        pfr_ne = np.zeros(len(pfr_pts)) + self.pfr_ne_val
        pfr_Ti = np.zeros(len(pfr_pts)) + self.pfr_Ti_val
        pfr_Te = np.zeros(len(pfr_pts)) + self.pfr_Te_val
        
        pts_ni_pfr = np.column_stack((pfr_pts,pfr_ni))
        pts_ne_pfr = np.column_stack((pfr_pts,pfr_ne))
        pts_Ti_pfr = np.column_stack((pfr_pts,pfr_Ti))
        pts_Te_pfr = np.column_stack((pfr_pts,pfr_Te))
        
        self.ni_pts     = np.vstack((self.ni_pts,pts_ni_pfr))
        self.ne_pts     = np.vstack((self.ne_pts,pts_ne_pfr))
        self.Ti_kev_pts = np.vstack((self.Ti_kev_pts,pts_Ti_pfr))
        self.Te_kev_pts = np.vstack((self.Te_kev_pts,pts_Te_pfr))
        
        #grid_x, grid_y = np.mgrid[1:2.5:500j, -1.5:1.5:500j]
        #ni_for_plot = griddata(self.ni_pts[:,:-1],self.ni_pts[:,-1],(grid_x,grid_y))
        #Ti_for_plot = griddata(self.Ti_kev_pts[:,:-1],self.Ti_kev_pts[:,-1],(grid_x,grid_y))
        #plt.contourf(grid_x,grid_y,np.log10(Ti_for_plot),500)
        #plt.colorbar()
        #sys.exit()
    
    def triangle_prep(self):

        sol_pol_pts = self.core_pol_pts + self.ib_div_pol_pts + self.ob_div_pol_pts
        
        #GET POINTS FOR TRIANGULATION
        #main seperatrix
        sep_pts = np.zeros((self.core_pol_pts,2))
        for i,v in enumerate(np.linspace(0,1,self.core_pol_pts,endpoint=False)):
            sep_pts[i] = np.asarray(self.main_sep_line.interpolate(v,normalized=True).xy).T[0]
        
        #inboard divertor leg
        ib_div_pts = np.zeros((self.ib_div_pol_pts,2))
        for i,v in enumerate(np.linspace(0,1,self.ib_div_pol_pts,endpoint=True)): #skipping the x-point (point 0)
            ib_div_pts[i] = np.asarray(self.ib_div_line_cut.interpolate(v,normalized=True).xy).T[0]
        
        #outboard divertor leg
        ob_div_pts = np.zeros((self.ob_div_pol_pts,2))
        for i,v in enumerate(np.linspace(0,1,self.ob_div_pol_pts,endpoint=True)): #skipping the x-point (point 0)
            ob_div_pts[i] = np.asarray(self.ob_div_line_cut.interpolate(v,normalized=True).xy).T[0]
                
        #core
        core_pts = np.zeros((self.core_pol_pts*len(self.core_lines),2))
        for num,line in enumerate(self.core_lines):
            for i,v in enumerate(np.linspace(0,1,self.core_pol_pts,endpoint=False)):
                core_pts[num*self.core_pol_pts + i] = np.asarray(line.interpolate(v,normalized=True).xy).T[0]
            
        self.core_ring = LinearRing(core_pts[:self.core_pol_pts])
        
        
        #sol
        sol_pts = np.zeros((sol_pol_pts*len(self.sol_lines_cut),2))
        for num,line in enumerate(self.sol_lines_cut):
            for i,v in enumerate(np.linspace(0,1,sol_pol_pts,endpoint=True)):
                sol_pts[num*sol_pol_pts + i] = np.asarray(line.interpolate(v,normalized=True).xy).T[0]         

        #wall        
        wall_pts = np.asarray(self.wall_line.coords)[:-1]
        self.wall_ring = LinearRing(wall_pts)

        all_pts = np.vstack((sep_pts,
                             ib_div_pts,
                             ob_div_pts,
                             core_pts,
                             sol_pts,
                             wall_pts))
        
        #CREATE SEGMENTS FOR TRIANGULATION
        #WHEN DOING WALL, CHECK EACH POINT TO SEE IF IT HAS ALREADY BEEN
        #CREATED. IF SO, USE THE NUMBER OF THAT POINT AND DELETE THE WALL
        #VERSION OF IT IN THE ALL_PTS ARRAY.

        sep_segs    = np.column_stack((np.arange(self.core_pol_pts),
                                       np.roll(np.arange(self.core_pol_pts),-1)))
        
        ib_div_segs = np.column_stack((np.arange(self.ib_div_pol_pts),
                                       np.roll(np.arange(self.ib_div_pol_pts),-1)))[:-1]
        
        ob_div_segs = np.column_stack((np.arange(self.ob_div_pol_pts),
                                       np.roll(np.arange(self.ob_div_pol_pts),-1)))[:-1]

        core_segs   = np.zeros((0,2),dtype='int')
        for i,v in enumerate(self.core_lines):
            new_segs = np.column_stack((np.arange(self.core_pol_pts),
                                        np.roll(np.arange(self.core_pol_pts),-1))) \
                                        + self.core_pol_pts * i
            core_segs = np.vstack((core_segs,new_segs))

        sol_segs    = np.zeros((0,2),dtype='int')
        for i,v in enumerate(self.sol_lines):
            new_segs = np.column_stack((np.arange(sol_pol_pts),
                                        np.roll(np.arange(sol_pol_pts),-1)))[:-1] \
                                        + sol_pol_pts * i
            sol_segs = np.vstack((sol_segs,new_segs))
        
        wall_segs   = np.column_stack((np.arange(len(wall_pts)),
                                       np.roll(np.arange(len(wall_pts)),-1)))
        
        all_segs = np.vstack((sep_segs,
                              ib_div_segs + len(sep_segs),
                              ob_div_segs + len(ib_div_segs) + len(sep_segs) + 1,
                              core_segs + len(ob_div_segs) + len(ib_div_segs) + len(sep_segs) + 1 + 1,
                              sol_segs + len(core_segs) + len(ob_div_segs) + len(ib_div_segs) + len(sep_segs)  + 1 + 1,
                              wall_segs + len(sol_segs) + len(core_segs) + len(ob_div_segs) + len(ib_div_segs) + len(sep_segs) + 1 + 1 + self.num_sollines
                              ))

        all_pts_unique = np.unique(all_pts,axis=0)

        #CLEANUP
        #NOTE: this process will result in a segments array that looks fairly chaotic,
        #but will ensure that the triangulation goes smoothly.
        
        #loop over each point in all_segs
        #look up the point's coordinates in all_pts
        #find the location of those coordinates in all_pts_unique
        #put that location in the corresponding location in all_segs_unique
        
        all_segs_unique = np.zeros(all_segs.flatten().shape,dtype='int')
        for i,pt in enumerate(all_segs.flatten()):
            pt_coords = all_pts[pt]
            loc_unique = np.where((all_pts_unique==pt_coords).all(axis=1))[0][0]
            all_segs_unique[i] = loc_unique
        all_segs_unique = all_segs_unique.reshape(-1,2)  

        ## OUTPUT .poly FILE AND RUN TRIANGLE PROGRAM
        open('exp_mesh.poly', 'w').close()
        outfile = open('exp_mesh.poly','ab')
        filepath = os.path.realpath(outfile.name)
        np.savetxt(outfile,
                   np.array([all_pts_unique.shape[0],2,0,0])[None],
                   fmt='%i %i %i %i')
        np.savetxt(outfile,
                   np.column_stack((np.arange(len(all_pts_unique)),
                                    all_pts_unique)),
                   fmt='%i %f %f')
        np.savetxt(outfile,
                   np.array([all_segs_unique.shape[0],0])[None],
                   fmt='%i %i')
        np.savetxt(outfile,
                   np.column_stack((np.arange(len(all_segs_unique)),
                                    all_segs_unique,
                                    np.zeros(len(all_segs_unique),dtype='int'))),
                   fmt='%i %i %i %i')
        np.savetxt(outfile,
                   np.array([1])[None],
                   fmt='%i')
        np.savetxt(outfile,
                   np.array([1,self.m_axis[0],self.m_axis[1]])[None],
                   fmt='%i %f %f')
        np.savetxt(outfile,
                   np.array([0])[None],
                   fmt='%i')
        outfile.close()
        
        #construct options to pass to triangle, as specified in input file
        #refer to https://www.cs.cmu.edu/~quake/triangle.html
        
        tri_min_angle = 10.0
        tri_min_area = 0.005
        tri_options = '-p'
        try:
            tri_options = tri_options + 'q' + str(tri_min_angle)
        except:
            pass
        
        try:
            tri_options = tri_options + 'a' + str(tri_min_area)
        except:
            pass
        
        tri_options = tri_options + 'nz'
        #call triangle
        try:
            call(['triangle', tri_options,filepath])
        except AttributeError:
            try:
                call(['triangle', tri_options,filepath])
            except:
                print 'triangle could not be found. Stopping.'
                sys.exit
        
    def read_triangle(self):
        ## READ TRIANGLE OUTPUT

        ## DECLARE FILE PATHS
        nodepath = os.getcwd() + '/exp_mesh.1.node'
        elepath = os.getcwd() + '/exp_mesh.1.ele'
        neighpath = os.getcwd() + '/exp_mesh.1.neigh'

        ## GET NODE DATA
        with open(nodepath, 'r') as node:
            #dummy = next(mil_mesh)
            nodecount = re.findall(r'\d+', next(node))
            nNodes = int(nodecount[0])
            nodenum = np.zeros(nNodes)
            nodesx = np.zeros(nNodes)
            nodesy = np.zeros(nNodes)
            
            for i in range (0,nNodes):
                data1 = re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', next(node)) 
                nodenum[i] = int(data1[0])
                nodesx[i] = data1[1]
                nodesy[i] = data1[2]
                
        ## GET TRIANGLE DATA
        with open(elepath, 'r') as tri_file:
            tricount = re.findall(r'\d+', next(tri_file))
            nTri = int(tricount[0])
            print 'number of triangles = ',nTri
            triangles = np.zeros((nTri,3))
            tri_regions = np.zeros(nTri)
            for i in range (0,nTri):
                data1 = re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', next(tri_file))
                triangles[i,0] = data1[1]
                triangles[i,1] = data1[2]
                triangles[i,2] = data1[3]
                #tri_regions[i] = data1[4]
        triangles = triangles.astype('int')
        tri_regions = tri_regions.astype('int')

        ## GET NEIGHBOR DATA
        with open(neighpath, 'r') as neigh_file:
            neighcount = re.findall(r'\d+', next(neigh_file))
            nNeigh = int(neighcount[0])
            neighbors = np.zeros((nNeigh,3))
            for i in range (0,nNeigh):
                data1 = re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', next(neigh_file))
                neighbors[i,0] = data1[1]
                neighbors[i,1] = data1[2]
                neighbors[i,2] = data1[3]
        neighbors = neighbors.astype('int')

        ## REARRANGE TRIANGLES TO CONFORM TO GTNEUT CONVENTION      
        triangles = np.fliplr(triangles) #triangle vertices are given counterclockwise, but we want clockwise
        neighbors = np.fliplr(neighbors) #neighbor 1 is opposite vertex 1, so also counterclockwise 
        
        y=np.zeros(3)
        for i,tri in enumerate(triangles):
            # Find lowest value of y component of vertices
            y[0] = nodesy[tri[0]]
            y[1] = nodesy[tri[1]]
            y[2] = nodesy[tri[2]]
            miny = np.amin(y)
            miny_count = np.sum(y == miny)
            if miny_count == 1:
                #identify position of minimum and roll array accordingly
                miny_index = np.where(y==miny)[0][0]
            else:
                #identify which points are the two minima and determine
                #which of them is farthest to the left (or right if I change it)
                miny_index = np.where(y==miny)[0][1] #change this 1 to a zero to choose the rightmost of the two bottom vertices
            triangles[i] = np.roll(triangles[i],-1*miny_index)
            neighbors[i] = np.roll(neighbors[i],-1*miny_index-2) # the -2 is because the side 1 is opposite vertex 1. We want side 1 to start at vertex 1
        
        ## GET VALUES TO ORIENT THE FIRST CELL WHEN PLOTTING
        point1_x = nodesx[triangles[0,0]]
        point1_y = nodesy[triangles[0,0]]
        point2_x = nodesx[triangles[0,1]]
        point2_y = nodesy[triangles[0,1]]
        point3_x = nodesx[triangles[0,2]]
        point3_y = nodesy[triangles[0,2]]
        
        cell1_ctr_x = (point1_x + point2_x + point3_x) / 3
        cell1_ctr_y = (point1_y + point2_y + point3_y) / 3

        ## CALCULATE ANGLE BY WHICH TO ROTATE THE FIRST CELL WHEN PLOTTING
        cell1_theta0 = degrees(self.getangle([point3_x,point3_y],[point1_x,point1_y]))    

        ## GET VALUES TO ORIENT THE FIRST CELL WHEN PLOTTING
        point1_x = nodesx[triangles[0,0]]
        point1_y = nodesy[triangles[0,0]]
        point2_x = nodesx[triangles[0,1]]
        point2_y = nodesy[triangles[0,1]]
        point3_x = nodesx[triangles[0,2]]
        point3_y = nodesy[triangles[0,2]]
        
        cell1_ctr_x = (point1_x + point2_x + point3_x) / 3
        cell1_ctr_y = (point1_y + point2_y + point3_y) / 3

        ## CALCULATE ANGLE BY WHICH TO ROTATE THE FIRST CELL WHEN PLOTTING
        cell1_theta0 = degrees(self.getangle([point3_x,point3_y],[point1_x,point1_y]))

        ## CALCULATE MID POINTS OF TRIANGLES, AS WELL AS MIDPOINTS FOR EACH FACE     
        ptsx = np.zeros((nTri,3))
        ptsy = np.zeros((nTri,3))
        #for index,tri in ndenumerate(triangles):
        for i in range(0,nTri):
            ptsx[i,0] = nodesx[triangles[i,0]]
            ptsy[i,0] = nodesy[triangles[i,0]]
            ptsx[i,1] = nodesx[triangles[i,1]]
            ptsy[i,1] = nodesy[triangles[i,1]]
            ptsx[i,2] = nodesx[triangles[i,2]]
            ptsy[i,2] = nodesy[triangles[i,2]]
        
        mid_x = np.mean(ptsx,axis=1)
        mid_y = np.mean(ptsy,axis=1)
        self.midpts = np.column_stack((mid_x,mid_y))
        
        side1_midx = (ptsx[:,0] + ptsx[:,1])/2
        side2_midx = (ptsx[:,1] + ptsx[:,2])/2
        side3_midx = (ptsx[:,2] + ptsx[:,0])/2
        
        side1_midy = (ptsy[:,0] + ptsy[:,1])/2
        side2_midy = (ptsy[:,1] + ptsy[:,2])/2
        side3_midy = (ptsy[:,2] + ptsy[:,0])/2
        
        side1_midpt = np.column_stack((side1_midx,side1_midy))
        side2_midpt = np.column_stack((side2_midx,side2_midy))
        side3_midpt = np.column_stack((side3_midx,side3_midy))
        
        #COMBINE POINTS FOR THE PLASMA, SOL, AND DIVERTOR REGIONS
        #first fill in plasma cells
        plasmacells = np.zeros((1,2))
        pcellnum = nTri
        pcellcount = 0
        
        for index,nei in enumerate(neighbors):
            #for each face of the cell, find the mid-point and check if it falls in line
            side1inline = self.isinline(side1_midpt[index],self.core_ring)
            side2inline = self.isinline(side2_midpt[index],self.core_ring)
            side3inline = self.isinline(side3_midpt[index],self.core_ring)

            if side1inline or side2inline or side3inline:
                nb = (nei == -1).sum() #count number of times -1 occurs in nei
                #print 'nb = ',nb
                #print side1_midpt[index]
                #print side2_midpt[index]
                #print side3_midpt[index]
                if nb == 1: #cell has one plasma border
                    
                    #create plasma cell
                    plasmacells[pcellcount,0] = pcellnum
                    plasmacells[pcellcount,1] = index
                    plasmacells = np.vstack((plasmacells,[0,0]))
                    #update neighbors
                    nei[np.argmax(nei==-1)] = pcellnum
                    #get ready for next run
                    pcellnum +=1
                    pcellcount +=1
                elif nb == 2: 
                    #cell has two plasma borders (this will probably never happen. It would require a local 
                    #concavity in the inner-most meshed flux surface)
                    #create plasma cell #1
                    plasmacells[pcellcount,0] = pcellnum
                    plasmacells[pcellcount,1] = index
                    plasmacells = np.vstack((plasmacells,[0,0]))
                    #update neighbors
                    nei[np.argmax(nei==-1)] = pcellnum
                    #get ready for next run
                    pcellnum +=1
                    pcellcount +=1   
                    
                    #create plasma cell #2
                    plasmacells[pcellcount,0] = pcellnum
                    plasmacells[pcellcount,1] = index
                    plasmacells = np.vstack((plasmacells,[0,0]))
                    #update neighbors
                    nei[np.argmax(nei==-1)] = pcellnum
                    #get ready for next run
                    pcellnum +=1
                    pcellcount +=1
        plasmacells = np.delete(plasmacells,-1,0)
        plasmacells = plasmacells.astype('int')

        #now fill in wall cells
        wallcells = np.zeros((1,6))
        wcellnum = pcellnum #was already advanced in the plasmacell loop. Don't add 1.
        wcellcount = 0
        
        print np.asarray(self.wall_ring.xy).T
        plt.scatter(np.asarray(self.wall_ring.xy).T[:,0],np.asarray(self.wall_ring.xy).T[:,1],marker='o',s=0.5,color='red')
        
        for index, nei in enumerate(neighbors):
            #for each face of the cell, find the mid-point and check if it falls in line
            side1inline = self.isinline(side1_midpt[index],self.wall_ring)
            side2inline = self.isinline(side2_midpt[index],self.wall_ring)
            side3inline = self.isinline(side3_midpt[index],self.wall_ring)
            
            if side1inline or side2inline or side3inline:
                print index,nei,side1inline,side2inline,side3inline
                nb = (nei == -1).sum() #count number of times -1 occurs in nei
                if nb == 1: #cell has one wall border
                    #identify the side that is the wall cell
                    sidenum = np.where(np.asarray([side1inline,side2inline,side3inline]))[0][0]
                    if sidenum==0:
                        pt = side1_midpt[index]
                    elif sidenum==1:
                        pt = side2_midpt[index]
                    elif sidenum==2:
                        pt = side3_midpt[index]
                    
                    #create wall cell
                    wallcells[wcellcount,0] = wcellnum
                    wallcells[wcellcount,1] = index
                    wallcells[wcellcount,2] = griddata(self.ni_pts[:,:-1],self.ni_pts[:,-1],pt,method='nearest',rescale=True)
                    wallcells[wcellcount,3] = griddata(self.ne_pts[:,:-1],self.ne_pts[:,-1],pt,method='nearest',rescale=True)
                    wallcells[wcellcount,4] = griddata(self.Ti_kev_pts[:,:-1],self.Ti_kev_pts[:,-1],pt,method='nearest',rescale=True)
                    wallcells[wcellcount,5] = griddata(self.Te_kev_pts[:,:-1],self.Te_kev_pts[:,-1],pt,method='nearest',rescale=True)
                    wallcells = np.vstack((wallcells,[0,0,0,0,0,0]))
                    #update neighbors
                    nei[np.argmax(nei==-1)] = wcellnum
                    #get ready for next run
                    wcellnum +=1
                    wcellcount +=1
                elif nb == 2: #cell has two wall borders (This can easily happen because the wall has many concave points.)
                    #create wall cell #1
                    wallcells[wcellcount,0] = wcellnum
                    wallcells[wcellcount,1] = index
                    wallcells = np.vstack((wallcells,[0,0,0,0,0,0]))
                    #update neighbors
                    nei[np.argmax(nei==-1)] = wcellnum
                    #get ready for next run
                    wcellnum +=1
                    wcellcount +=1   
                    
                    #create wall cell #2
                    wallcells[wcellcount,0] = wcellnum
                    wallcells[wcellcount,1] = index
                    wallcells = np.vstack((wallcells,[0,0,0,0,0,0]))
                    #update neighbors
                    nei[np.argmax(nei==-1)] = wcellnum
                    #get ready for next run
                    wcellnum +=1
                    wcellcount +=1
        wallcells = np.delete(wallcells,-1,0)
        wallcells = wallcells.astype('int')
        sys.exit()
        ## POPULATE CELL DENSITIES AND TEMPERATURES
        #create array of all points in plasma, sol, id, and od
        #tri_param = np.vstack((plasma_param,sol_param,id_param,od_param))
        
        ni_tri = griddata(self.ni_pts[:,:-1],
                          self.ni_pts[:,-1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)
        ne_tri = griddata(self.ne_pts[:,:-1],
                          self.ne_pts[:,-1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)
        Ti_tri = griddata(self.Ti_kev_pts[:,:-1],
                          self.Ti_kev_pts[:,-1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)
        Te_tri = griddata(self.Te_kev_pts[:,:-1],
                          self.Te_kev_pts[:,-1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)

        #ni_tri[ni_tri<1.0E16] = 1.0E16
        #ne_tri[ne_tri<1.0E16] = 1.0E16
        #Ti_tri[Ti_tri<0.002] = 0.002
        #Te_tri[Te_tri<0.002] = 0.002
        
        ## CALCULATE LENGTHS OF SIDES
        lsides = np.zeros((nTri,3))
        for i in range (0,nTri):
            lsides[i,0] = sqrt((ptsx[i,0]-ptsx[i,1])**2 + (ptsy[i,0]-ptsy[i,1])**2)
            lsides[i,1] = sqrt((ptsx[i,1]-ptsx[i,2])**2 + (ptsy[i,1]-ptsy[i,2])**2)
            lsides[i,2] = sqrt((ptsx[i,2]-ptsx[i,0])**2 + (ptsy[i,2]-ptsy[i,0])**2)
        
        ## CALCULATE CELL ANGLES
        angles = np.zeros((nTri,3))
        for i in range (0,nTri):
            p1 = np.array([ptsx[i,0],ptsy[i,0]])
            p2 = np.array([ptsx[i,1],ptsy[i,1]])
            p3 = np.array([ptsx[i,2],ptsy[i,2]])
            angles[i,0] = self.getangle3ptsdeg(p1,p2,p3)
            angles[i,1] = self.getangle3ptsdeg(p2,p3,p1)
            angles[i,2] = self.getangle3ptsdeg(p3,p1,p2)
        
        ## WRITE NEUTPY INPUT FILE!    
        f = open(os.getcwd() + '/neutpy_in_generated','w')
        f.write('nCells = ' + str(nTri) + ' nPlasmReg = ' + str(pcellcount) + ' nWallSegm = ' + str(wcellcount))
        for i in range(0,nTri):
            f.write('\n'+'iType(' + str(i) + ') = 0 nSides(' + str(i) + ') = 3 ' + 'adjCell('+str(i)+') = '+', '.join(map(str, neighbors[i,:])))
        f.write('\n')
        f.write('\n#lsides and angles for normal cells')
        for i in range(0,nTri):
            f.write('\n'+'lsides(' + str(i) + ') = '+', '.join(map(str, lsides[i,:]))+' angles(' + str(i) + ') = '+', '.join(map(str, angles[i,:])))
        f.write('\n')
        f.write('\n#densities and temperatures for normal cells')
        for i in range(0,nTri):
            f.write('\n'+'elecTemp('+str(i)+') = '+str(Te_tri[i]) +' elecDens(' + str(i) + ') = '+str(ne_tri[i])+' ionTemp('+str(i)+') = '+str(Ti_tri[i]) +' ionDens(' + str(i) + ') = '+str(ni_tri[i]))
        f.write('\n')
        f.write('\n#wall cells')
        for i,wcell in enumerate(wallcells):
            f.write('\n'+'iType('+str(wcell[0])+') = 2 nSides('+str(wcell[0])+') = 1 adjCell('+str(wcell[0])+') = '+str(wcell[1])+' zwall('+str(wcell[0])+') = 6 awall('+str(wcell[0])+') = 12 twall('+str(wcell[0])+') = '+str(wcell[4])+' f_abs('+str(wcell[0])+') = 0.0 s_ext('+str(wcell[0])+') = 1.0E19') 
        f.write('\n')
        f.write('\n#plasma core and vacuum cells')
        for i,pcell in enumerate(plasmacells):
            f.write('\n'+'iType(' + str(pcell[0]) + ') = 1 nSides(' + str(pcell[0]) + ') = 1 adjCell(1, ' + str(pcell[0]) + ') = ' + str(pcell[1]) + ' twall(' + str(pcell[0]) + ') = 5000  alb_s(' + str(pcell[0]) + ') = 0  alb_t(' + str(pcell[0]) + ') = 0  s_ext(' + str(pcell[0]) + ') = 0 ')
        f.write('\n')
        f.write('\n#general parameters')
        f.write('\nzion = 1 ')
        f.write('\naion = 2 ')
        f.write('\naneut = 2 ')
        f.write('\ntslow = 0.002 ')
        f.write('\n')
        f.write('\n#cross section and reflection model parameters')
        f.write('\nxsec_ioni = janev')
        f.write('\nxsec_ione = janev')
        f.write('\nxsec_cx = janev')
        f.write('\nxsec_el = janev')
        f.write('\nxsec_eln = stacey_thomas')
        f.write('\nxsec_rec = stacey_thomas')
        f.write('\nrefmod_e = stacey')
        f.write('\nrefmod_n = stacey')
        f.write('\n')
        f.write('\n#transmission coefficient parameters')
        f.write('\nint_method = midpoint')
        f.write('\nphi_int_pts = 10')
        f.write('\nxi_int_pts = 10')
        f.write('\n')
        f.write('\n#make a bickley-naylor interpolated lookup file. (y or n)')
        f.write('\nmake_bn_int = n')
        f.write('\n')
        f.write('\n#extra (optional) arguments for plotting')
        f.write('\ncell1_ctr_x  = ' + str(cell1_ctr_x))
        f.write('\ncell1_ctr_y  = ' + str(cell1_ctr_y))
        f.write('\ncell1_theta0 = ' + str(cell1_theta0))
        f.write('\n')
        f.close()

        
        #create dictionary to pass to neutpy
        toneutpy={}
        toneutpy["nCells"]       = nTri
        toneutpy["nPlasmReg"]    = pcellcount
        toneutpy["nWallSegm"]    = wcellcount
        toneutpy["aneut"]        = 2
        toneutpy["zion"]         = 1
        toneutpy["aion"]         = 2
        toneutpy["tslow"]        = 0.002
        toneutpy["int_method"]   = 'midpoint'
        toneutpy["phi_int_pts"]  = 10
        toneutpy["xi_int_pts"]   = 10
        toneutpy["xsec_ioni"]    = 'janev'
        toneutpy["xsec_ione"]    = 'janev'
        toneutpy["xsec_cx"]      = 'janev'
        toneutpy["xsec_rec"]     = 'stacey_thomas'
        toneutpy["xsec_el"]      = 'janev'
        toneutpy["xsec_eln"]     = 'stacey_thomas'
        toneutpy["refmod_e"]     = 'stacey'
        toneutpy["refmod_n"]     = 'stacey'
        
        toneutpy["iType"]        = np.asarray([0]*nTri + [1]*pcellcount + [2]*wcellcount)
        toneutpy["nSides"]       = np.asarray([3]*nTri + [1]*(pcellcount + wcellcount))
        toneutpy["zwall"]        = np.asarray([0]*(nTri+pcellcount) + [6]*wcellcount)
        toneutpy["awall"]        = np.asarray([0]*(nTri+pcellcount) + [12]*wcellcount)
        toneutpy["elecTemp"]     = Te_tri[:nTri]
        toneutpy["ionTemp"]      = Ti_tri[:nTri]
        toneutpy["elecDens"]     = ne_tri[:nTri]
        toneutpy["ionDens"]      = ni_tri[:nTri]
        toneutpy["twall"]        = np.asarray([0]*nTri + [5000]*pcellcount + [0.002]*wcellcount)
        toneutpy["f_abs"]        = np.asarray([0]*(nTri+pcellcount) + [0]*wcellcount)
        toneutpy["alb_s"]        = np.asarray([0]*nTri + [0]*pcellcount + [0]*wcellcount)
        toneutpy["alb_t"]        = np.asarray([0]*nTri + [0]*pcellcount + [0]*wcellcount)
        toneutpy["s_ext"]        = np.asarray([0.0]*nTri + [0.0]*pcellcount + [0.0]*wcellcount)
        
        toneutpy["adjCell"]      = neighbors
        toneutpy["lsides"]       = lsides
        toneutpy["angles"]       = angles
        toneutpy["cell1_ctr_x"]  = cell1_ctr_x
        toneutpy["cell1_ctr_y"]  = cell1_ctr_y
        toneutpy["cell1_theta0"] = cell1_theta0
        
        time0 = time.time()
        self.neutpy_inst = neutpy(inarrs=toneutpy)
        time1 = time.time()
        print 'neutpy time = ',time1-time0
        plot = neutpyplot(self.neutpy_inst)
        self.nn_s_raw = self.neutpy_inst.cell_nn_s
        self.nn_t_raw = self.neutpy_inst.cell_nn_t
        self.nn_raw = self.nn_s_raw + self.nn_t_raw
        
        self.iznrate_s_raw = self.neutpy_inst.cell_izn_rate_s
        self.iznrate_t_raw = self.neutpy_inst.cell_izn_rate_t
        self.iznrate_raw = self.iznrate_s_raw + self.iznrate_t_raw
        
        #create output file
        #the file contains R,Z coordinates and then the values of several calculated parameters
        #at each of those points.
        
        f = open(self.neutfile_loc,'w')
        f.write(('{:^18s}'*8).format('R','Z','n_n_slow','n_n_thermal','n_n_total','izn_rate_slow','izn_rate_thermal','izn_rate_total'))
        for i,pt in enumerate(self.midpts):
            f.write(('\n'+'{:>18.5f}'*2+'{:>18.5E}'*6).format(
                                        self.midpts[i,0],
                                        self.midpts[i,1],
                                        self.nn_s_raw[i],
                                        self.nn_t_raw[i],
                                        self.nn_raw[i],
                                        self.iznrate_s_raw[i],
                                        self.iznrate_t_raw[i],
                                        self.iznrate_raw[i]))
        f.close()        


inst = neutpy_prep('toneutprep')

"""
fig_width = 6.0
fig_height = (np.amax(inst.Z) - np.amin(inst.Z)) / (np.amax(inst.R) - np.amin(inst.R)) * fig_width

fig1 = plt.figure(figsize=(0.975*fig_width,fig_height))
ax1 = fig1.add_subplot(1,1,1)
ax1.axis('equal')

#psi raw
#ax1.imshow(np.flipud(inst.psi),aspect='auto')

#ax1.contourf(inst.R,inst.Z,inst.psi,500)

Br = np.gradient(inst.psi,axis=1)/inst.R
Bz = -np.gradient(inst.psi,axis=0)/inst.R
B_p = np.sqrt((np.gradient(inst.psi,axis=1)/inst.R)**2 + \
                (-np.gradient(inst.psi,axis=0)/inst.R)**2)
ax1.contourf(inst.R,inst.Z,B_p,500)

ax1.plot(inst.wallx,inst.wally,color='black',lw=1,zorder=10)

for i in range(int(len(inst.dpsidR_0)/2)):
    x1,y1 = np.split(inst.dpsidR_0[i],2,axis=1)
    #ax1.plot(x1,y1,color='red',lw=1,label = 'dpsidR=0')
for i in range(int(len(inst.dpsidZ_0)/2)):
    x1,y1 = np.split(inst.dpsidZ_0[i],2,axis=1)
    #ax1.plot(x1,y1,color='blue',lw=1,label = 'dpsidZ=0')
#ax1.legend()

#xpt and mag_axis
#ax1.scatter(inst.xpt[0],inst.xpt[1],color='yellow',zorder=10)
#ax1.scatter(inst.m_axis[0],inst.m_axis[1],color='yellow',zorder=10)

#core density
#R,Z = np.meshgrid(np.linspace(1.02,2.31,500),np.linspace(-1.15,0.95,500))
#ni = griddata(inst.ni_pts[:,:-1],inst.ni_pts[:,2],(R,Z),method='cubic',fill_value=inst.ni_pts[-1,2]*1.0)
#dnidr = np.abs(np.gradient(ni,axis=1)) + np.abs(np.gradient(ni,axis=0))
#ax1.contourf(R,Z,dnidr,500)


#seperatrix
coords = np.asarray(inst.main_sep_line.coords)
coords = np.vstack((coords,coords[0]))
ax1.plot(coords[:,0],coords[:,1],color='yellow',lw=1)

#inboard divertor
coords = np.asarray(inst.ib_div_line_cut.coords)
ax1.plot(coords[:,0],coords[:,1],color='yellow',lw=1)

#outboard divertor
coords = np.asarray(inst.ob_div_line_cut.coords)
ax1.plot(coords[:,0],coords[:,1],color='yellow',lw=1)

#plot core lines
for i,line in enumerate(inst.core_lines):
    coords = np.asarray(line.coords)
    coords = np.vstack((coords,coords[0]))
    #ax1.plot(coords[:,0],coords[:,1],color='pink',lw=1)

#plot sol lines
for i,line in enumerate(inst.sol_lines_cut):
    coords = np.asarray(line.coords)
    #ax1.plot(coords[:,0],coords[:,1],color='lime',lw=1)
"""
