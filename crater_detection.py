# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:56:50 2021

@author: Hugo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import ReadIM
import scipy.integrate as scpint
from scipy.optimize import curve_fit
import scipy.signal as scpsign
from numpy.polynomial.legendre import Legendre
import os
import pickle



class mixing_layer_detection():
    def __init__(self, path, load):
        self.path=os.path.abspath(path)
        self.files=self.importation(self.path)        
        self.img_surface=self.build_img('surface.im7')
        self.img_surface=cv2.convertScaleAbs(self.img_surface,alpha=(255.0/4095.0))
        self.img_background=self.build_img('background.im7')
        
        if load:
            self.iteration=True
            self.load()
        else:
            self.iteration=False
            self.mark_surface()
            self.list_sym_axis(self.files)
            self.iterate(self.files)
            self.save()

    def importation(self,path):       
        files=[]  
        files = [os.path.join(path,f) for f in os.listdir(path) if f[-4:]=='.im7']
        
        return files

    def build_img(self,file):
        vbuff, vatts = ReadIM.extra.get_Buffer_andAttributeList(file)
        v_array, vbuff = ReadIM.extra.buffer_as_array(vbuff)
        del(vbuff)
        return v_array[0]

    def crop(self,img,z_surf):
        img= img[z_surf:,:]
        return img
    
    def remove_background(self,img,background):
        img= (4095.0-background)+img  
        return img

    def threshold(self,img):
        img=cv2.convertScaleAbs(img,alpha=(255.0/4095.0)) 
        ret,img = cv2.threshold(img,245,255,cv2.THRESH_TRUNC) 
        return img
        
    def img_processing(self,img,coordinate='Cartesian'):        
        img=self.remove_background(img,self.img_background)
        img=self.crop(img,self.z_surf)
        img=self.threshold(img)     
        a=self.sym_axis(img)
        value=img.shape[0]
        polar_image = cv2.warpPolar(img,np.shape(img),(a,0), value, cv2.WARP_FILL_OUTLIERS)
        if coordinate=='Cartesian':
            return img
        elif coordinate=='Polar':
            l=int(len(polar_image)/2)
            return polar_image[:l,:]
    
    def blur(self,img):
        imgCopy = cv2.blur(img,(11,11))
        #imgCopy=cv2.GaussianBlur(img,(51,51),0)
        #imgCopy = cv2.bilateralFilter(img,9,75,75)
        return imgCopy 
    
    def gradient(self,img,blur):
        if blur:
           img=self.blur(img)
        gx, gy = np.gradient(img)
        return np.sqrt(gx**2+gy**2)

    def click_event(self,event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_surf, self.z_surf=(x, y)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img_surface, str(x) + ',' + str(y), (x,y), font,1, (0, 255, 0), 2)
            cv2.imshow('image', self.img_surface)
            
    def mark_surface(self):
        cv2.imshow('image', self.img_surface)
        cv2.setMouseCallback('image', self.click_event)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def detect_ext_layer(self,img):
        h,w =np.shape(img)
        layer=[]
        for i in range(h):
            col=-img[i,:] 
            col=col- np.min(col) #+img[-1,:]
            cons_samples=10
            threshold=5
            bin_array=np.convolve(col< threshold, np.ones((cons_samples)), 'same') == cons_samples
            first_1=np.argmax(bin_array)
            layer.append(first_1)
        return layer
            
    def detect_int_layer(self,grad, ext_lay):
        h,w =np.shape(grad)
        layer=[]
        for i in range(h):
            col=grad[i,:]
            col=scpsign.savgol_filter(col,51,5)
            peaks=scpsign.find_peaks_cwt(col,np.arange(5,20))
            peaks=np.flip(peaks)
            if len(peaks)>0:
                for j in range(len(peaks)):
                    if peaks[j]<=ext_lay[i] and j<len(peaks)-1:
                        layer.append(peaks[j+1])
                        break
                    if j==len(peaks)-1:
                        layer.append(peaks[j])
                    
            else:
                layer.append(0)
                    
        return layer  

            
    def noise_rmv(self,int_lay,ext_lay):
        """Enleve le bruit hors du cratere"""
        n=len(int_lay)
        for i in range(n):
            if int_lay[i]>ext_lay[i]:
                int_lay[i]=0
        return int_lay
    
    # def layer_thickness(self,ext_lay,int_lay):
    #     x=np.arange(len(ext_lay))
    #     thickness=scpint.simps(ext_lay,x)-scpint.simps(int_lay,x)
    #     return thickness
                
    def iterate(self,files):
        self.ext_layer=[]
        self.int_layer=[]
        self.iteration=True
        for file in files:
            img=self.build_img(file)
            img=self.img_processing(img,coordinate='Polar')
            grad=self.gradient(img,blur='True')
            ext_lay=self.detect_ext_layer(img)
            self.ext_layer.append(ext_lay)
            int_lay=self.detect_int_layer(grad,ext_lay)
            int_lay=self.noise_rmv(int_lay,ext_lay)
            self.int_layer.append(int_lay)
        self.ext_layer=np.array(self.ext_layer)
        self.int_layer=np.array(self.int_layer)
        self.ext_layer=scpsign.savgol_filter(self.ext_layer,51,5)
        self.int_layer=scpsign.savgol_filter(self.int_layer,51,5)


    
    def plot_layer(self,i,coordinate='Cartesian',legendre=True):
        if self.iteration:
            if coordinate=='Polar':
                img=self.build_img(self.files[i])
                img=self.img_processing(img,coordinate='Polar')
                img=np.transpose(img)
                int_layer_filtered=scpsign.savgol_filter(self.int_layer,51,5)
                plt.imshow(img,cmap='gray')
                plt.plot(self.ext_layer[i])
                plt.plot(int_layer_filtered[i])
                plt.show()
            elif coordinate=='Cartesian':
                img=self.build_img(self.files[i])
                img=self.img_processing(img,coordinate='Cartesian')
                #img=np.transpose(img)
                int_layer_filtered=scpsign.savgol_filter(self.int_layer[i],51,5)
                th=np.linspace(0,np.pi,len(int_layer_filtered))
                ext_layer=self.ext_layer[i]
                x_ext=ext_layer*np.cos(th)+self.axis[i]
                x_int=int_layer_filtered*np.cos(th)+self.axis[i]
                plt.imshow(img,cmap='gray')
                if legendre:
                        r_ext, r_int=p.legendre_fit(i)
                        a,b,c=r_ext
                        R_ext=p.radius_legendre2(th,a,b,c)
                        x_ext_fit=R_ext*np.cos(th)+self.axis[i]
                        y_ext_fit=R_ext*np.sin(th)  
                        a,b,c=r_int
                        R_int=p.radius_legendre2(th,a,b,c)
                        x_int_fit=R_int*np.cos(th)+self.axis[i]
                        y_int_fit=R_int*np.sin(th)
                        plt.plot(x_ext_fit,y_ext_fit)
                        plt.plot(x_int_fit,y_int_fit)
                    

                plt.plot(x_ext,ext_layer*np.sin(th))
                plt.plot(x_int,int_layer_filtered*np.sin(th))
                plt.show()                
        else:
            print('Error : iterate must be run first')
            
    
    
    def polar_plot(self,i):
        img=self.build_img(self.files[i])
        img=self.img_processing(img)    
        self.a=self.sym_axis(img)
        #value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
        value=img.shape[0]
        polar_image = cv2.warpPolar(img,np.shape(img),(self.a,0), value, cv2.WARP_FILL_OUTLIERS)        
        polar_image = polar_image.astype(np.uint8)
        plt.imshow(polar_image)
        plt.show()
        
    def radius_legendre4(self,theta,a,b,c,d,e):
        A=[a,b,c,d,e]
        R=Legendre(A)
        x=2*np.cos(theta)-1
        return R(x)
    
    def radius_legendre2(self,theta,a,b,c):
        A=[a,b,c]
        R=Legendre(A)
        x=2*np.cos(theta)-1
        return R(x)
            
    def legendre_fit(self,i):
        if self.iteration:
            N=len(self.ext_layer[i])
            theta=np.arange(len(self.ext_layer[i]))
            theta=np.linspace(0,np.pi,N) ###
            popt_ext, pcov = curve_fit(self.radius_legendre2,theta,self.ext_layer[i] )
            popt_int, pcov = curve_fit(self.radius_legendre2,theta,self.int_layer[i] )
            return popt_ext, popt_int
        else:
            print('Error : iterate must be run first')
        
    
    def chronophotographe(self,files, i):
        cpg=[]
        for file in files:
            img=self.build_img(file)
            img=self.img_processing(img)
            cpg.append(img[:,i])
            cpg.append(img[:,i])
        return np.transpose(cpg)
    
    def sym_axis(self, img):
        img=255-img
        img=scpsign.savgol_filter(img,51,1)
        img_mirror=np.flip(img.copy(),axis=1)
        convolution=[]
        for i in range(len(img)):
            conv=scpsign.correlate(img[i,:],img_mirror[i,:])
            convolution.append(conv)
        a= np.unravel_index(np.argmax(convolution),np.shape(convolution))
        return a[1]/2
    
    def list_sym_axis(self,files):
        self.axis=[]
        for file in files:
            img=self.build_img(file)
            img=self.img_processing(img)
            a=self.sym_axis(img)
            self.axis.append(a)
        
    
    def mixing_layer_thickness(self):
        theta=np.arange(len(p.ext_layer[0]))
        thickness=[]
        for i in range(len(self.ext_layer)):
            popt_ext, popt_int=self.legendre_fit(i)
            a,b,c=popt_ext
            R_ext=self.radius_legendre2(theta,a,b,c)
            a,b,c=popt_int
            R_int=self.radius_legendre2(theta,a,b,c)
            thickness.append(np.mean(R_ext)-np.mean(R_int))
            
        return thickness
    
    def save(self):
        data={"ext_layer":self.ext_layer,"int_layer":self.int_layer, "axis":self.axis, "surface":self.z_surf}
        pickle.dump(data,open( "save.p", "wb" ) )
        
    def load(self):
        #self.iteration=True
        data=pickle.load( open( "save.p", "rb" ) )
        self.ext_layer=data["ext_layer"]
        self.int_layer=data["int_layer"]
        self.axis=data["axis"]
        self.z_surf=data["surface"]
        
            
    def show_images(self):
        global img
        global win_name
        self.modif_ext={}
        self.modif_int={}
        win_name='Images'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 960, 540)
        cv2.moveWindow(win_name, 0, 0)
        i=100
        th=np.linspace(0,np.pi,len(p.ext_layer[i]))
        x_ext=self.ext_layer[i]*np.cos(th)+self.axis[i]
        x_int=self.int_layer[i]*np.cos(th)+self.axis[i] 
        y_ext=self.ext_layer[i]*np.sin(th)
        y_int=self.int_layer[i]*np.sin(th)  
        ext_lay=np.array([[x_ext[k],y_ext[k]] for k in range(len(x_ext))],np.int32)
        ext_lay= ext_lay.reshape((-1, 1, 2))
        int_lay=np.array([[x_int[k],y_int[k]] for k in range(len(x_ext))],np.int32)
        int_lay= int_lay.reshape((-1, 1, 2))
        img=self.build_img(self.files[i])
        img=self.img_processing(img,coordinate='Cartesian')
        img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        thickness=5
        isClosed=True
        cv2.polylines(img,ext_lay, isClosed, (255, 0, 0), thickness)
        cv2.polylines(img, int_lay, isClosed, (0, 0, 255), thickness)
        cv2.imshow(win_name, img)
        while True:
            key = cv2.waitKey(0)
            if key == 27: # 27 = esc
                cv2.destroyAllWindows()
                print('exit')
                break
            if key == ord('q'):
                i-=1
                img=self.build_img(self.files[i])
                img=self.img_processing(img,coordinate='Cartesian')
                img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                x_ext=self.ext_layer[i]*np.cos(th)+self.axis[i]
                x_int=self.int_layer[i]*np.cos(th)+self.axis[i]
                y_ext=self.ext_layer[i]*np.sin(th)
                y_int=self.int_layer[i]*np.sin(th)               
                ext_lay=np.array([[x_ext[k],y_ext[k]] for k in range(len(x_ext))],np.int32)
                ext_lay= ext_lay.reshape((-1, 1, 2))
                int_lay=np.array([[x_int[k],y_int[k]] for k in range(len(x_ext))],np.int32)
                int_lay= int_lay.reshape((-1, 1, 2))
                cv2.polylines(img, ext_lay, isClosed, (255, 0, 0), thickness)
                cv2.polylines(img, int_lay, isClosed, (0, 0, 255), thickness)
                cv2.imshow(win_name, img)
                print('previous image')
            if key == ord('d'):
                i+=1
                img=self.build_img(self.files[i])
                img=self.img_processing(img,coordinate='Cartesian')
                img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                x_ext=self.ext_layer[i]*np.cos(th)+self.axis[i]
                x_int=self.int_layer[i]*np.cos(th)+self.axis[i]               
                y_ext=self.ext_layer[i]*np.sin(th)
                y_int=self.int_layer[i]*np.sin(th)                
                ext_lay=np.array([[x_ext[k],y_ext[k]] for k in range(len(x_ext))],np.int32)
                ext_lay= ext_lay.reshape((-1, 1, 2))
                int_lay=np.array([[x_int[k],y_int[k]] for k in range(len(x_ext))],np.int32)
                int_lay= int_lay.reshape((-1, 1, 2))
                cv2.polylines(img, ext_lay, isClosed, (255, 0, 0), thickness)
                cv2.polylines(img, int_lay, isClosed, (0, 0, 255), thickness)
                cv2.imshow(win_name, img)
                print('next image')
            if key == ord('z'): #modification ext
                font = cv2.FONT_HERSHEY_SIMPLEX
                img=self.build_img(self.files[i])
                img=self.img_processing(img,coordinate='Cartesian')
                img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.putText(img, 'Modification couche externe', (100,1000), font,2, (0, 0, 0), thickness)
                cv2.imshow(win_name, img)
                mouseX,mouseY=(0,0)
                self.new_ext_lay=[]
                while True:
                    cv2.setMouseCallback(win_name, self.point_surface_ext) 
                    print((mouseX,mouseY))
                    key2=cv2.waitKey(0)
                    if key2==27:
                        print('exit2')
                        self.new_ext_lay.pop(0)
                        self.new_ext_lay=np.array(self.new_ext_lay)
                        break  
                self.modif_ext[i]=self.new_ext_lay
            if key == ord('s'): #modification int
                font = cv2.FONT_HERSHEY_SIMPLEX
                img=self.build_img(self.files[i])
                img=self.img_processing(img,coordinate='Cartesian')
                img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.putText(img, 'Modification couche interne', (100,1000), font,2, (0, 0, 0), thickness)
                cv2.imshow(win_name, img)
                mouseX,mouseY=(0,0)
                self.new_int_lay=[]
                while True:
                    cv2.setMouseCallback(win_name, self.point_surface_int)                     
                    print((mouseX,mouseY))
                    key2=cv2.waitKey(0)
                    if key2==27:
                        print('exit2')
                        self.new_int_lay.pop(0)
                        self.new_int_lay=np.array(self.new_int_lay)
                        break
                self.modif_int[i]=self.new_int_lay
                
    def point_surface_int(self,event, x, y, flags, params):
        global mouseX , mouseY
        thickness=10
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY= (x,y)
            self.new_int_lay.append([np.sqrt(mouseX**2+mouseY**2),np.arctan2(mouseY,mouseX)])
            cv2.circle(img,(mouseX,mouseY),5,(0,0,255),thickness)
            cv2.imshow(win_name, img)

    def point_surface_ext(self,event, x, y, flags, params):
        global mouseX , mouseY
        thickness=10
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY= (x,y)
            self.new_ext_lay.append([np.sqrt(mouseX**2+mouseY**2),np.arctan2(mouseY,mouseX)])
            cv2.circle(img,(mouseX,mouseY),5,(0,0,255),thickness)
            cv2.imshow(win_name, img)        
        
# =============================================================================
               
if __name__ == "__main__":
    path='E:\Documents\ENS\M2\Stage\image_processing\\0405\\NaCl_25cm_115\\data'
    #path='E:\Documents\ENS\M2\Stage\image_processing\\0405\\NaCl_25cm_Date=210504_2'
    load=True
    p=mixing_layer_detection(path,load)
    #p.iterate(p.files)
    #p.save()
    p.load()
    chro=p.chronophotographe(p.files,1200)
    plt.imshow(chro)
    plt.show()
    
    l=p.mixing_layer_thickness()
    plt.plot(l)
    plt.show()
    
    
 
                    
        
