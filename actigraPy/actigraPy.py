#!/usr/bin/python
"""

read and do some processig on actiwatch AWD files


JWE Jan 2019

"""

import numpy as np
import os, sys
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import zscore
from copy import deepcopy
import csv
import tkinter as tk
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg

"""
maybe make a global var for dat?
"""
dt_fmt = '%d-%b-%y %I:%M %p'

def read_dat(fn_pref):
    """
    need to add the column names or something, need to add headers, also add marker idx as dict to output?
    """

    #with open(fn_pref + '_dat.csv','r') as fid:
    #    reader = csv.reader(fid)
    #    #csv_reader = csv.DictReader(csv_file)
    #    dat = list(reader)   
    
    dat = pd.read_csv(fn_pref + '_dat.csv',keep_default_na=False)
    #dat = dat.get_values()   # this may not be the best

    # get the idx for the markers from the dat file
    r,c = np.shape(dat)
    mks = list(dat.columns[3:])
    #print(c)
    marker_idx = {}
    #for ii in np.arange(c-2)+2:
    for mm in mks:
        #tmp = [ idx for idx,val in enumerate(dat[mm].values) if val=='1' ]  #was mm
        tmp = np.where(np.diff(dat[mm].astype(int)))[0] + 1
        #print(tmp)
        marker_idx[mm] = tmp

    return dat,marker_idx

def get_idx(dat_time,mk_times,pos=False):
    """
    from a data file with times and marker times in the same format, return indices for all the markers
    """
    
    dat_time = list(dat_time)
    if isinstance(dat_time[0],str):
        dt_time = [ dt.datetime.strptime(ii,dt_fmt) for ii in dat_time ]
        str_time = dat_time
    else:
        dt_time = dat_time
        str_time = [ dt.datetime.strftime(ii,dt_fmt) for ii in dat_time ]
    mk_len = len(mk_times)
    # initialize the mk_idx with only comments
    mk_idx = [[None,None,ii[2]] for ii in mk_times]
    exist_list = [[False,False]]*mk_len
    for idx,(st,en,com) in enumerate(mk_times):
        # check start that the st,en times are within range and get index, otherwise
        # put the min, max of the range, respectively
        for time_idx,time in enumerate([st,en]):
            if isinstance(time,str):
               time = dt.datetime.strptime(time,dt_fmt)
            ok = True
            if time  < dt_time[0]:
               mk_idx[idx][time_idx] = 0
               ok = False
            if time  > dt_time[-1]:
               mk_idx[idx][time_idx] = len(dat_time)-1
               ok = False
            if ok:
               mk_idx[idx][time_idx] = dt_time.index(time)
               exist_list[idx][time_idx]= True
   
    mk_idx = [ tuple(ii) for ii in mk_idx ]
    exist_list = [ tuple(ii) for ii in exist_list ]
    if pos:
        return mk_idx,exist_list
    else:
        return mk_idx
    
def read_log(fn,awd_dat={}):

    dt_fmt = '%d-%b-%y %I:%M %p'

    fn_pref,fn_ext = os.path.splitext(fn)
    if fn_ext == '.csv':
        log_dat = pd.read_csv(fn, keep_default_na=False)
    elif fn_ext == '.xls':
        log_dat = pd.read_excel(fn)

    #print(log_dat)

    # check for comment column and extract the rows that have the keywords
    # this will fail miserably if there is no Comment column...
    keywords = {'watch_on':['Start','start'],'watch_off':['End','end']}
    kw_dat = {}
    for ii in keywords.keys():
       val = np.where(log_dat.Comment.isin(keywords[ii]))[0]
       #print(ii,val)
       if len(val) > 0:
          kw_dat[ii] = (log_dat.iloc[val])
          log_dat.drop(log_dat.index[val],inplace=True)

    log_dat['On'] = pd.to_datetime(log_dat['OnDate'].astype(str) +
                                   ' ' + log_dat['OnTime'].astype(str) )
    log_dat['Off'] = pd.to_datetime(log_dat['OffDate'].astype(str) + 
                                    ' ' + log_dat['OffTime'].astype(str) )
    #log_dat.drop(['OnDate','OnTime','OffDate','OffTime'],inplace=True)
    log_dat['Comment']=log_dat['Comment'].fillna("")
    log_dat = log_dat.to_dict(orient='list')    


    if fn_ext == '.csv':
        st_time = np.array([ ' '.join(ii) for ii in zip(log_dat['OffDate'],log_dat['OffTime'])])
        en_time = np.array([ ' '.join(ii) for ii in zip(log_dat['OnDate'],log_dat['OnTime'])])

    elif fn_ext == '.xls':

        st_time = np.array([ dt.datetime.strftime(ii,dt_fmt) for ii in log_dat['Off'] ])
        en_time = np.array([ dt.datetime.strftime(ii,dt_fmt) for ii in log_dat['On'] ])

        # get the on off times
        # also currently assumes kw_dat has length =2...
        if 'watch_on' in kw_dat.keys():
           log_dat['watch_on'] =  dt.datetime.strftime(dt.datetime.combine(kw_dat['watch_on']['OnDate'].dt.date.values[0],kw_dat['watch_on']['OnTime'].values[0]),dt_fmt)
        if 'watch_off' in kw_dat.keys():
           log_dat['watch_off'] =  dt.datetime.strftime(dt.datetime.combine(kw_dat['watch_off']['OffDate'].dt.date.values[0],kw_dat['watch_off']['OffTime'].values[0]),dt_fmt)

    #mk_time = [ val for pair in zip(st_time, en_time) for val in pair]

    # get data indices (either read file or use read_marker?), return indices

    #print(dat[0].values,mk_time)


    #mk_idx, pos =  get_idx(awd_dat['DateTime'],mk_time,pos=True)
    if 'marker' in log_dat.keys():
        um = np.unique(log_dat['marker'])
        # check for comment markers remove from list
        mk_dict = {}
        for mm in um:
             mm_log_idx = np.where(np.array(log_dat['marker']) == mm)[0]
             x = st_time[mm_log_idx]
             y = en_time[mm_log_idx]
             z = np.array(log_dat['Comment'])[mm_log_idx]
             mm_time = list(zip(x,y,z))
             mm_idx, pos =  get_idx(awd_dat['DateTime'],mm_time,pos=True)
             rm_list=[]
             for idx,tup in enumerate(pos):
                 if tup == (False,False):
                      rm_list.append(idx)
             rm_list.sort(reverse=True)
             for idx in rm_list:
                 mm_idx.pop(idx)

             mk_dict[mm]=mm_idx
        log_dat['mks'] = mk_dict 
        #log_dat['idx'] =  mk_idx
        log_dat['idx'] =  []
        
    else:
        mk_list = list(zip(st_time,en_time,np.array(log_dat['Comment'])))
        #mk_idx, pos =  get_idx(awd_dat['DateTime'],mk_time,pos=True)
        mk_idx,pos=get_idx(awd_dat['dt_list'],mk_list,pos=True)
        #adjust for comment blocks that are totally out of range of AWD
        rm_list=[]
        for idx in range(0,len(mk_idx)):
            if pos[idx]==(False,False):
                rm_list.append(idx)
        rm_list.sort(reverse=True)
        for idx in rm_list:
            mk_idx.pop(idx)

        log_dat['idx']=mk_idx

    return log_dat,kw_dat

def write_edited(fn_pref,dat=[],hdr=[],mk_idx=[]):

    # get orig data (assumes the script was already run, but then how else 
    # would they have the /MMtimes file.
    
    if not hdr:
       orig_awd = read_AWD(fn_pref + '.AWD')
    if not dat:
       orig_dat,orig_m_idx = read_dat(fn_pref)
    if not mk_idx:
       Mtimes,mk_idx = read_marker(fn_pref)

    #now get list of markers
    # if it's a dict of many types, combine into one
    if type(mk_idx) == dict:
        all_tup = []
        for mm in mk_idx.keys():
            mm_tup = [ (val,mk_idx[mm][2*ii+1]) for ii,val in enumerate(mk_idx[mm][:-1:2]) ]
            all_tup = all_tup + mm_tup
        
        all_tup.sort(key=lambda tup: tup[0])
        all_mk_idx = [ ii for jj in all_tup for ii in jj]


    m_marks = [''] * len(orig_dat['activity'])
    for ii in all_mk_idx: m_marks[ii]='M'

    #fn_pref = dat_fn.split('_')[0]
    print(fn_pref)
    rows = zip(orig_dat['activity'],m_marks)
    with open(fn_pref + '_edited.AWD', "w") as f:
        writer = csv.writer(f,delimiter=' ')
        for row in awd_dat['hdr']:
           writer.writerow([row])
        for row in rows:
            writer.writerow(row)
    
def write_dat(awd_dat,mk_idx,fn_pref,fn_suff=''):

   dat = { ii: awd_dat[ii] for ii in ['DateTime','activity','M'] }
   dat = pd.DataFrame.from_dict(dat)

   # make marker dicts into full length series
   for mm in mk_idx.keys():
      mm_marks = [0] * len(dat)
      for block in mk_idx[mm]:
         idx1 = block[0]
         idx2= block[1]
         mm_marks[idx1:idx2]= [1]*(idx2-idx1)
      
      #print(mm_marks)
      tmp = pd.DataFrame({mm:mm_marks})
      dat = dat.join(tmp)

   dat.to_csv(fn_pref + '_' +  fn_suff + 'dat.csv',index=False)

   return dat

def find_nearest(array, value):
    """
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    from:
    https://matplotlib.org/gallery/user_interfaces/embedding_in_tk_canvas_sgskip.html
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo

def code_act(idat):

   # use the actiwatch weighting for minute intervals a = 0.04 (e-2) + 0.2 (e-1) + e + 0.2(e+1) + 0.04 (e+2)

   act = np.zeros(len(idat)+2)
   # can only really start coding at the 3rd datapoint, and 2 from end...a
   act_weight = [0.04,0.2,1,0.2,0.04]
   for idx,val in enumerate(idat[2:-2]):
       act[idx+2] = np.sum(idat[idx:idx+5] * act_weight)
      
   # pad the points that can't be coded...
   act[0:2] = [act[2]]*2
   #act[-2:] = [act[-2]]*2
   act = act[:-2]

   # code activity thresholds
   a40 = act >= 40
   a20 = act >= 20

   #a40 = [ 'a40' if ii else '' for ii in a40[0] ]
   #a20 = [ 'a20' if ii else '' for ii in a20[0] ]

   return act,a40,a20

def clip_dat(lim,awd_dat):

   # get all the same length data from the dict
   dat = { ii: awd_dat[ii] for ii in ['DateTime','activity', 'dt_list','M'] }
   dat = pd.DataFrame.from_dict(dat)

   tmp = get_idx(awd_dat['DateTime'],lim)
   [st,en]=[tmp[0][0],tmp[0][1]]
   clipped_dat =  dat.iloc[st:en,:]
   clipped_dat = clipped_dat.to_dict(orient='list')

   clipped_dat['N'] = len(clipped_dat['activity'])
   clipped_dat['hdr'] = awd_dat['hdr']

   return clipped_dat

def despike(dat,zlev=4,win=2):
   # start with simple cutoff and replace with median or something.
   #dat_mean = np.mean(dat)
   #dat_std = np.std(dat)
   dat = np.array(dat)
   zap  = zscore(dat)
   ozap = np.array(abs(zap)>zlev)
    
   out_ind = ozap.nonzero()
   out_val = dat.compress(ozap)
    
   # now replace them with some sort of average over their n nearest neigbours
   nap = deepcopy(dat)
   for jj in out_ind[0]:
      if jj + win > dat.size:
         ewin = dat.size
      else:
         ewin = int(jj + win)
 
      if jj - win < 0:
         swin = 0
      else:
         swin = int(jj - win)
      #print(jj,ewin,swin,dat[swin:jj])
      nap[jj] = np.mean(np.concatenate([dat[swin:jj],dat[int(jj+1):ewin]]))
 
   return nap

def plot_awd(awd_dat,mk_idx,plot_type='single',show=True,fn_pref='',max_act=-1,debug=False):
#def plot_awd(DateTime,idat,list_idx,plot_type='single',comments=[],show=True,fn_pref='',max_act=-1,debug=False):
   idat = awd_dat['activity']

   # want to plot activity data by day
   # get number of days, then split data by days, the bar plota

   #idat = np.array([ int(ii) for ii in dat if ii is not('M') ] ) 

   day_list = [ ii.split()[0] for ii in awd_dat['DateTime'] ]
   days = list(np.unique(day_list))
   days.sort(key=lambda x: dt.datetime.strptime(x, '%d-%b-%y'))
   #print(days)
   n_days = len(days)
   #n_days =  14
   #days = days[:n_days]
 
   print(n_days)
     
   wh = round(n_days/7)*8

   if plot_type == 'double':
      ww = 12
   else:
      ww = 6

   awd_fig = plt.figure(facecolor='w',figsize=(ww,wh))
   awd_fig.clear()

   #for dd,day in enumerate(days[:28]):
   #for dd,day in enumerate(days[28:35]):
   for dd,day in enumerate(days):
      print(day)
      #ax = awd_fig.add_subplot(n_days,1,dd+1)
      ax = awd_fig.add_subplot(n_days,1,dd+1)
      # get data that matches (you really only have to do this for the first day (and last sort of) because it should be 1440 rows per full day
      dd_idx = [idx for idx,ddd  in enumerate(day_list) if ddd == day]
      if not max_act > 0:
         max_act = max(awd_dat['activity'])
      ax.set_ylim([0,max_act])
      comment_height = max_act/2

      min_idx = np.min(dd_idx)

      max_idx = np.max(dd_idx)
      if plot_type == 'double':
         tmp = max_idx + 1440
         if tmp < len(idat): 
            max_idx = max_idx + 1440
         else:
            max_idx = len(idat)
      offset =0
      delt_idx = max_idx-min_idx # this should just be the number of days plotted in a row...

      if dd==0:
          offset = 1439-max_idx
          delt_idx = 1439
   
      
      if debug:
        print(min_idx,max_idx)

      #tmp = idat[dd_idx]
      plt.bar(np.arange(offset,delt_idx),idat[min_idx:max_idx],width=1)
     
      colours = ['blue','red','darkred','pink','lightcyan']
      for cc,mm in enumerate(mk_idx.keys()):
          mk_dd_idx= np.where([(x[1]>min_idx)&(x[0]<max_idx) for x in mk_idx[mm]])[0].tolist()
          mk_dd = []
          for idx in mk_dd_idx:
              tup = mk_idx[mm][idx]
              [st,en]=[tup[0],tup[1]]
              if tup[0]<min_idx:
                  st=min_idx
              if tup[1]>max_idx:
                  en=max_idx
              tmp=(st,en,tup[2])
              mk_dd.append(tmp)
          for idx,ii in enumerate(mk_dd):
              x1=np.abs(ii[0])-min_idx+offset
              x2=np.abs(ii[1])-min_idx +offset
              print("plotting: %d,%d from %s"%(x1,x2,mm))
              ax.axvspan(x1,x2, alpha=0.3, color=colours[np.mod(cc,len(colours))])
              jitter = (cc % 2) * comment_height // 4
              ax.text(x1,comment_height+jitter,ii[2])

            #for mm in M_idx[m_idx]:
            #   ax.text(mm-min_idx,idat[mm],'M')
               #print(mm)

      ax.set_ylabel(day)
      ax.set_xticks(np.arange(0,1439,60))
      if plot_type=='double':
         ax.set_xticklabels(list(np.arange(24))*2)
      else:
         ax.set_xticklabels(np.arange(24))
      #ax.axis('off')
      ax.spines["top"].set_visible(False)
      ax.spines["right"].set_visible(False)
      ax.spines["bottom"].set_visible(False)

   # legend, do once for all keys
   all_patch = [] # for legend
   for cc,mm in enumerate(mk_idx.keys()):
      mm_patch = mpatches.Patch(color=colours[np.mod(cc,len(colours))], label=mm,alpha=0.3)
      all_patch.append(mm_patch) 
      plt.legend(handles=all_patch)

   plt.tight_layout()

   #if not fn_pref: fn_pref = 'test'
   if fn_pref:
      print('Saving... ' + fn_pref + '.png')
      awd_fig.savefig(fn_pref + '.png', bbox_inches='tight')
      print('Done')

   if show:
      root=tk.Tk()
      fw = ww*100
      fh = 800
      frame=tk.Frame(root,width=fw,height=fh)
      frame.grid(row=0,column=0)
      canvas=tk.Canvas(frame,bg='#FFFFFF',width=fw,height=fh,scrollregion=(0,0,fw,4800))  #600?
      #hbar=tk.Scrollbar(frame,orient=tk.HORIZONTAL)
      #hbar.pack(side=tk.BOTTOM,fill=tk.X)
   #  #hbar.config(command=canvas.xview)
      vbar=tk.Scrollbar(frame,orient=tk.VERTICAL)
      vbar.pack(side=tk.RIGHT,fill=tk.Y)
      vbar.config(command=canvas.yview)
      canvas.config(width=fw,height=fh)
      canvas.config(yscrollcommand=vbar.set)
      canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
   
      fig_x, fig_y = 0,0
      fig_photo = draw_figure(canvas, awd_fig, loc=(fig_x, fig_y))
      fig_w, fig_h = fig_photo.width(), fig_photo.height()
   
      #canvas.postscript(file="tmp.ps", colormode="color")
   
      root.mainloop()

def read_AWD(fn):
   
   awd_dat = {}  # make this global?

   hdr_keys=['sub','start_date','start_time','unknown','age','watch','gender'] 
   n_hdr = len(hdr_keys)

   with open(fn, 'r') as fid:
        dat = []
        M = []
        for line in fid:
            p = line.split()
            if len(p) > 1:
               dat.append(p[0])
               M.append(p[1])
            else:
               dat.append(p[0])
               M.append('')

   fid.close()
   
   hdr = dat[0:n_hdr]
   awd_dat['hdr'] = dict(zip(hdr_keys, hdr))

   dat = dat[n_hdr:]
   awd_dat['activity'] = np.array(dat,dtype=int)
   awd_dat['M'] = M[n_hdr:]

   # get start dt
   #start_dt = dt.datetime.strptime(' '.join([hdr['start_date'],hdr['start_time']]),'%d-%b-%Y %H:%M')
   start_dt = dt.datetime.strptime(' '.join([hdr[1],hdr[2]]),'%d-%b-%Y %H:%M')
   

   # create list of dts by increment of a minute to end of data
   inc = dt.timedelta(minutes=1)

   awd_dat['N'] = len(dat)
   awd_dat['dt_list'] = [ start_dt + ii*inc for ii in np.arange(awd_dat['N']) ]
   awd_dat['DateTime'] = [ ii.strftime('%d-%b-%y %I:%M %p') for ii in awd_dat['dt_list'] ]

   return awd_dat


def write_Mtimes(awd_dat,mk_idx,fn_pref,fn_suff=''):
   dt_fmt = "%d-%b-%y %I:%M %p"
   mk_list = []

   for mm in mk_idx.keys():
       if mm != 'M':
            for tup in mk_idx[mm]:
                mk_list.append(tup+(mm,))
   mk_list.sort()
   dat = pd.DataFrame()

   dat['Off'] = [awd_dat['dt_list'][x[0]].strftime(dt_fmt) for x in mk_list]
   dat['On'] = [awd_dat['dt_list'][x[1]].strftime(dt_fmt) for x in mk_list]


   for ii in ['On','Off']:
       tmp = dat[ii].str.split(" ", n = 1, expand = True)
       dat[ii + 'Date'] = tmp[0]
       dat[ ii + 'Time'] = tmp[1]
   del dat['On']
   del dat['Off']

   dat['marker'] = [x[3] for x in mk_list]
   dat['Comment'] = [x[2] for x in mk_list]
   dat.to_csv(fn_pref + '_Mtimes'+fn_suff+'.csv', sep=',',index=False)

   return dat
def get_markers(awd_dat,log_fn=[]):
 
   N = awd_dat['N']
   dat = awd_dat['activity']
   fn_pref = awd_dat['hdr']['sub']
   if not log_fn:   # for back compatibility but can probably leave out
      log_fn = fn_pref + '_sleeplog.csv'

   # get markers indices, and number of markers
   M_idx = [ ii for ii,val in enumerate(awd_dat['M']) if val == 'M']

   idat = np.array(dat,dtype=int)
   # zero pad to prep for diff
   idat = np.concatenate([np.zeros(1,dtype=int),idat])

   btmp = idat > 0

   # get activity coded data
   act,a40,a20 = code_act(idat)
   ia40= np.concatenate([np.zeros(1,dtype=int),a40])

   bdat=ia40
   pdat=a40

   # and diff to find the transition points
   tp_tmp = np.diff(bdat.astype(int))

   tp_idx = np.where(tp_tmp)
   tp_idx = list(tp_idx[0])
 
   tp_idx = np.array(tp_idx)
   if tp_idx[0]:
      tp_idx = [ ii if np.mod(ii,2) else ii-1 for ii in tp_idx ]
      tp_idx = np.concatenate([np.zeros(1),tp_idx])
   if np.mod(len(tp_idx),2) == 0:
      tp_idx = np.concatenate([tp_idx,np.ones(1)*N-1])   # insert the last point

   tp_idx = np.array(tp_idx).astype(int)
   
   len_segs = np.diff(tp_idx)
   
   tmp = zip(tp_idx,tp_idx[1:])
   seg_idx  = [ ii for ii in tmp ]

   #print(len(seg_idx),len(tp_idx),tp_idx[0],tp_idx[-2:])

   # figure out if the start block is act/rest, zero/non-zero
   tmp = pdat[tp_idx[0]:tp_idx[1]]
   st_block = not(any(tmp))  # true if zero/not active

   #print(st_block,tmp,tp_idx[0],pdat[tp_idx[0]:tp_idx[2]])
  
   # remove short active segments, may wish to do this only for overnight periods?
   shorts = len_segs<5
   act_segs = [not(st_block),st_block]*round(len(seg_idx)/2) 
   # remove short act segments
   # for an even number of points there's an uneven number of segments...
   rem = shorts*act_segs[:len(shorts)]  # bad kludge
   

   # create a mask or binarized version of the data to avoid jiggery pokery relating to indices
   new_act_segs = [ act_segs[ii] if not val else False  for ii,val in enumerate(rem) ]
   x = [ [1]*len_segs[ii] if val else [0]*len_segs[ii] for ii,val in enumerate(new_act_segs) ]
   mask = [ jj for ii in x for jj in ii]

   # get the new transition points
   new_tp_idx = np.where(np.diff(mask))[0]

   if new_tp_idx[0]:
      new_tp_idx = [ ii if np.mod(ii,2) else ii-1 for ii in new_tp_idx ]
      new_tp_idx = np.concatenate([np.zeros(1),new_tp_idx])
   if np.mod(len(new_tp_idx),2) == 0:
      new_tp_idx = np.concatenate([new_tp_idx,np.ones(1)*N-1])   # insert the last point

   st_block = mask[0] == 0  # true if zero/not active, there is a better way to code this...

   #print(st_block,tmp,tp_idx[0],tp_idx[1],dat)
   
   # if there is a log file put in the markers and assume they are correct...
   wM_idx = deepcopy(M_idx)
   keep_idx = []
   if os.path.isfile(log_fn):
      log_dat,kw_dat = read_log(log_fn,awd_dat)
      mk_idx=[y for z in [x[0:2] for x in log_dat['idx']] for y in z]
      # all the log markers are 'right', if there's an M marker nearby, then use it for accuracy,and remove from the working list
      # if not, use the log
      if len(wM_idx)>0:
          th = 10  # use a more generous window?
          for ii,ll in enumerate(mk_idx):
             ll = np.abs(ll)
             match_idx_M,loc_idx = find_nearest(wM_idx,ll)
             #print(match_idx_M,ll)
             if np.abs(match_idx_M-ll) < th:
                #print(loc_idx)
                keep_idx.append(match_idx_M)
                del(wM_idx[loc_idx])   # remove it from the working M_idx list for below
             else:
                keep_idx.append(ll)
   else:
       log_dat = {}
   # modify the mask to inlcude the marked segments
   #xM = [ [1]*len_Msegs[ii] if val else [0]*len_Msegs[ii] for ii,val in enumerate(keep_idx) ]
   #maskM = [ jj for ii in xM for jj in ii]
 
   # get the new transition points  (this really should be a function)
   M_tp_idx = np.where(np.diff(mask))[0]

   if tp_idx[0]:
      tp_idx = [ ii if np.mod(ii,2) else ii-1 for ii in tp_idx ]
      tp_idx = np.concatenate([np.zeros(1),tp_idx])
   if np.mod(len(M_tp_idx),2) == 0:
      tp_idx = np.concatenate([tp_idx,np.ones(1)*N-1])   # insert the last point, there may be a lot of these?

   st_block = mask[0] == 0  # true if zero/not active, there is a better way to code this...

   th = 3
   mtp_idx = []
   # if there are remaining markers, see if they overlap with z
   if len(wM_idx) > 0:
      for mm in wM_idx:
         match_idx_tp,_ = find_nearest(tp_idx,mm)
         if np.abs(match_idx_tp-mm) < th:
            #print(mm,idx)
            mtp_idx.append(match_idx_tp)

   # of course, markers don't necessarily come in pairs 'cause that'd be too easy.
   # basically, there should be an even and odd tp_idx, if not add it.
   tmp_idx = []
   for ii in mtp_idx:
      tmp = np.where(tp_idx==ii)[0][0]
      eo = np.mod(tmp,2)
      tmp_idx.append(ii)
      if (eo == 0 and st_block==0) or ( eo == 1 and st_block >0 ) :  # may need to account for start act/non
         tmp_idx.append(tp_idx[tmp+1])
      else:
         if tmp != N-2:
            tmp_idx.append(tp_idx[tmp-1])

   tmp_idx = np.unique(tmp_idx)  # in case there are actually pairs remove the dups, yes, being lazy.

   keep_idx = keep_idx + list(tmp_idx)

   keep_idx.sort()  # really bad idea...  

   # the sort below should really be correct for keeping segment start/end pairs together
   # however, as overlapping M and z segments aren't currently checked the above sort is a
   # dirty workaround, which will probably create problems...
   
   sort_tmp = [ (val,keep_idx[2*ii+1]) for ii,val in enumerate(keep_idx[::2])]

   sort_tmp.sort(key=lambda tup: tup[0])
   keep_idx = [ int(ii) for jj in sort_tmp for ii in jj]


   # remove the matches from the tp_idx list so that the operations below can happen.
   # this seems inefficient but oh well...
#   tp_idx = list(tp_idx)
#   for ii in keep_idx:
#      try:
#         tmp = np.where(tp_idx==ii)[0][0]
#         del(tp_idx[tmp])
#      except:
#         pass

   # now deal with the unmarked segments ...
   #each point is a minutes so keep those segments that are lt a certain amount
   len_segs = np.diff(new_tp_idx)
   
   tmp = zip(new_tp_idx,new_tp_idx[1:])
   new_seg_idx  = [ ii for ii in tmp ]
   longs = len_segs>10
   #longs = len_segs>10

   nact_segs = [st_block,not(st_block)]*round(len(new_seg_idx)/2) 
   #nact_segs = [st_block,not(st_block)]*round(len(seg_idx)/2) 

   # keep segs in longs (t) and also non-act (t)
   keep = longs[:len(nact_segs)]*nact_segs
   z_segs = [ new_seg_idx[ii] for ii,val in enumerate(keep) if val ]
   #z_idx = [ int(ii)  for jj in z_segs for ii in jj]

   out_idx = {}
   M_list = list(zip(M_idx[0::2], M_idx[1::2]))
   out_idx ['M'] = [(x[0],x[1],"") for x in M_list] #talk to Jen about this -- pretty clumsy
   out_idx['z'] = [(int(x[0]),int(x[1]),"") for x in z_segs]
   out_idx['m'] = [(int(x[0]),int(x[1]),"") for x in sort_tmp]
   #if log_dat:
      #out_idx['l'] = np.sort(np.array(log_dat['idx']))  # sometimes they're not in order?
 
   return out_idx

if __name__ == "__main__":
   
   #fn = 'BCWM.AWD.csv'`
   fn = sys.argv[1]

   fn_base = os.path.basename(fn)
   fn_pref = os.path.splitext(fn_base)[0]

   awd_dat = read_AWD(fn)

   # despike by removing values 4 stdevs out
   dat=despike(awd_dat['activity'],4,10) 
   awd_dat['dat'] = dat

   out_idx,comments = get_markers(awd_dat)

   #dt_dat = list(zip(awd_dat['DateTime'],awd_dat['activity']))
   
   # so, what if there are no or few markers..
   # then binarize to get the transition points

   write_Mtimes(awd_dat,out_idx,fn_pref,comments)
   write_dat(awd_dat,out_idx,fn_pref)





