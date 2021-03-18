from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import column

def calc_moments(bins,weights):
	k = np.array([3,4])
	x = bins*weights
	z = (x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis]
	return x.mean(axis=1), (z**4).mean(axis = 1) - 3

class analysis():
	def __init__(self, obj, ID):
		self.ID = ID
		self.obj = obj
		self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}
		self.plt_color_bokeh = {'u':'magenta', 'g':'green', 'r':'red','i':'black','z':'blue'}
		self.marker_dict = {1:'s', 2:'v', 3:'o'}
		self.marker_dict_bokeh = {1:'square',2:'triangle',3:'circle'}
		self.survey_dict = {1:'SDSS', 2:'PS1', 3:'ZTF'}

	def read_in(self, reader, multi_proc = True, catalogue_of_properties = None, redshift=True):
		"""
		Read in raw data

		Parameters
		----------
		reader : function
				used for reading data
		multi_proc : boolean
				True to use multiprocessesing
		catalogue_of_properties : dataframe
		"""
		
		# Default to 4 cores
		pool = Pool(4)
		df_list = pool.map(reader, range(4))
		self.df = pd.concat(df_list).rename(columns={'mag_ps':'mag'})

		# Remove objects with a single observation.
		self.df = self.df[self.df.index.duplicated(keep=False)]
		if redshift:
			self.redshifts = pd.read_csv('data/catalogues/dr14q_uid_desig_z.csv', index_col=self.ID, usecols=[self.ID,'z'], squeeze=True).rename('redshift')
			self.df = self.df.join(self.redshifts, how = 'left', on=self.ID)
			self.df['mjd_rf'] = self.df['mjd']/(1+self.df['redshift'])
		
		self.coords = pd.read_csv('data/catalogues/dr14q_uid_desig_z.csv', index_col=self.ID, usecols=[self.ID,'ra','dec'])
		
# 		self.df = self.df.sort_index()
		assert self.df.index.is_monotonic, 'Index is not sorted'

	def residual(self, corrections):
		self.df = self.df.reset_index().set_index([self.ID,'catalogue'])
		for cat in corrections.keys():
			self.df.loc[pd.IndexSlice[:, cat], 'mag'] += corrections[cat]
		self.df = self.df.reset_index('catalogue')
		assert self.df.index.is_monotonic, 'Index is not sorted'

	def summary(self):
		"""
		Run to create the following attributes:
		self.idx_uid : array_like
				unique set of uids
		self.uids_missing : array_like
				uids of objects which are present in DR14Q but not in observations
		self.n_qsos : int
				number of objects for which we have observations
		self.idx_cat : array_like
				list of surveys which contribute to observations (sdss=1, ps=2, ztf=3)
		"""
		self.idx_uid	  = self.df.index.unique()
		uids_complete	 = pd.Index(np.arange(1,526356+1), dtype = np.uint32)
		self.uids_missing = uids_complete[~np.isin(uids_complete,self.idx_uid)]
		self.n_qsos		   = len(self.idx_uid)
		self.idx_cat	  = self.df['catalogue'].unique()

		print('Number of qsos with lightcurves: {:,}'.format(self.n_qsos))
		print('Number of datapoints in:\nSDSS: {:,}\nPS: {:,}\nZTF: {:,}'.format((self.df['catalogue']==1).sum(),
																				 (self.df['catalogue']==2).sum(),
																				 (self.df['catalogue']==3).sum()))
		
	def search(self, ra_dec, arcsec_threshold):
		"""
		Search database by ra and dec and return objects within threshold 

		Parameters
		----------
		ra_dec : numpy array
			list of ra and dec, with shape (N,2)
		arcsec_threshold : float
			measured in arcseconds
		Returns
		-------
		found : pandas.DataFrame of uid, ra, dec, distance of object to given ra, dec
			desc
		"""
		ra_dec = np.array(ra_dec).reshape((len(ra_dec),2))
		
		ra1 = self.coords['ra'].values
		dec1= self.coords['dec'].values
		ra2 = ra_dec[:,0,np.newaxis]
		dec2= ra_dec[:,1,np.newaxis]
		
		ra_sep = (ra1-ra2)*np.cos( np.deg2rad( 0.5*(dec1+dec2) ) )
		dec_sep= (dec1-dec2)
		dist = (ra_sep**2 + dec_sep**2)**0.5 * 3600
		boolean = (dist<1)
		offset  = dist[boolean]
		found   = self.coords[boolean.any(axis=0)]
		found['dist_arcsec'] = offset
		
		return found

			
	def group(self, keys = ['uid'], read_in = True, ztf = False):
		"""
		Group self.df by keys and apply {'mag':['mean','std','count'], 'magerr':'mean', 'mjd': ['min', 'max', np.ptp]}

		Add columns to self.df_grouped:
				redshift   : by joining vac['redshift'] along uid)
				mjd_ptp_rf : max ∆t in rest frame for given object (same as self.properties)

		Parameters
		----------
		keys : list of str
		read_in : boolean
		ztf : boolean
		"""
		#df_z = pd.read_csv('data/catalogues/dr14q_uid_desig_z.csv', usecols = [0,6], index_col = 0) if we need redshift
		if read_in == True:
			if len(keys) == 1:
				if ztf == True:
					self.df_grouped = pd.read_csv('data/surveys/ztf/meta_data/ztfdr2_gb_uid_{}.csv'.format(self.band),index_col = 0) #change this to ztf/gb.csv?
				else:
					self.df_grouped = pd.read_csv('data/merged/qsos/meta_data/df_gb_uid_{}.csv'.format(self.band),index_col = 0)
			elif len(keys) == 2:
				self.df_grouped = pd.read_csv('data/merged/qsos/meta_data/df_gb_uid_cat_{}.csv'.format(self.band),index_col = [0,1])
		elif read_in == False:
			# median_mag_fn/mean_mag_fn calculate mean/median magnitude by fluxes rather than mags themselves
			mean_mag_fn   = ('mean'  , lambda mag: -2.5*np.log10(np.mean  (10**(-(mag-8.9)/2.5))) + 8.9)
			median_mag_fn = ('median', lambda mag: -2.5*np.log10(np.median(10**(-(mag-8.9)/2.5))) + 8.9)

			self.df_grouped = self.df.groupby(keys).agg({'mag':[mean_mag_fn, median_mag_fn,'std','count'], 'magerr':'mean', 'mjd': ['min', 'max', np.ptp]})
			self.df_grouped.columns = ["_".join(x) for x in self.df_grouped.columns.ravel()]
			self.redshifts = pd.read_csv('data/catalogues/dr14q_uid_desig_z.csv', index_col=self.ID, usecols=[self.ID,'z'], squeeze=True).rename('redshift')

			self.df_grouped = self.df_grouped.merge(self.redshifts, on=self.ID)
		self.df_grouped['mjd_ptp_rf'] = self.df_grouped['mjd_ptp']/(1+self.df_grouped['redshift'])

	def load_vac(self):
		self.vac = pd.read_csv('data/catalogues/SDSS_DR12Q_BH_matched.csv', index_col=self.ID)
	
	def merge_with_catalogue(self,catalogue='dr12_vac', remove_outliers=True, prop_range_any = {'MBH_MgII':(6,12), 'MBH_CIV':(6,12)}):
		"""
		Reduce self.df to intersection of self.df and catalogue.
		Compute summary() to reupdate idx_uid, uids_missing, n_qsos and idx_cat
		Create new DataFrame, self.properties, which is inner join of [df_grouped, vac] along uid.
		Add columns to self.properties:
				mag_abs_mean : mean absolute magnitude
				mjd_ptp_rf   : max ∆t in rest frame for given object

		Parameters
		----------
		catalogue : DataFrame
				value added catalogue to be used for analysis
		remove_outliers : boolean
				remove objects which have values outside range specified in prop_range_any
		prop_range_any : dict
				dictionary of {property_name : (lower_bound, upper_bound)}

		"""
		if catalogue == 'dr12_vac':
			prop_range_all = {'Mi':(-30,-20),'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48),'nEdd':(-3,0.5)}
			self.prop_range = {**prop_range_all, **prop_range_any}
			cat = pd.read_csv('data/catalogues/SDSS_DR12Q_BH_matched.csv', index_col=self.ID)
			cat = cat.drop(columns='z')
# 			cat = cat.rename(columns={'z':'redshift'});

		self.df = self.df[self.df.index.isin(cat.index)]

		# Recalculate which qsos we are missing and which we have, given a list (copy of code in self.summary)
		self.summary()

#			   self.properties = self.df_grouped.reset_index('catalogue').join(vac, how = 'inner', on=self.ID)
		self.properties = self.df_grouped.join(cat, how = 'inner', on=self.ID)

		#calculate absolute magnitude
		self.properties['mag_abs_mean'] = self.properties['mag_mean'] - 5*np.log10(3.0/7.0*self.redshifts*(10**9))
		self.properties['mjd_ptp_rf']   = self.properties['mjd_ptp']/(1+self.redshifts)

		if remove_outliers==True:
			# Here, the last two entries of the prop_range dictionary are included on an any basis (ie if either are within the range)
			mask_all = np.array([(bound[0] < self.properties[key]) & (self.properties[key] < bound[1]) for key, bound in prop_range_all.items()])
			mask_any  = np.array([(bound[0] < self.properties[key]) & (self.properties[key] < bound[1]) for key, bound in prop_range_any.items()])
			mask = mask_all.all(axis=0) & mask_any.any(axis=0)
			self.properties = self.properties[mask]

	def bounds(self,key, bounds = np.array([-6,-1.5,-1,-0.5,0,0.5,1,1.5,6]), save=False):
		"""
		Compute z score of key for each object

		Parameters
		----------
		key : string
				property from VAC

		Returns
		-------
		bounds : array_like
				array of bounds to be used
		z_score : pandas.Series
				z value of property column (value-mean / std)
		self.bounds_values : values of property for each value in bounds
		ax : axes handle
		"""
		fig, ax = plt.subplots(1,1,figsize = (6,3))
		z_score = (self.properties[key]-self.properties[key].mean())/self.properties[key].std()
		z_score.hist(bins = 200, ax=ax)
		self.bounds_values = bounds * self.properties[key].std() + self.properties[key].mean()
		for i in range(len(bounds)-1):
#					   print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])&(self.properties['mag_count']>2)).sum()))
			print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])).sum()))
		for bound in bounds:
			ax.axvline(x=bound, color = 'k')
		ax.set(xlabel=key)
		if save == True:
			fig.savefig('bins_{}.pdf'.format(key),bbox_inches='tight')
		return bounds, z_score, self.bounds_values, ax


	def mean_drift():
		pass

	def bin_dtdm(self, uids=None, n_bins_t = 1000, n_bins_m = 200, t_max=7600, t_spacing = 'log', m_spacing = 'log', read_in = False, key = None, ztf=False, rest_frame=True):
		"""
		Take batch of qsos from a MBH bin. Calculate all dtdm for these qsos.
		Section these values into 19 large Δt bins with logarithmic spacing.
		Within these large bins, bin Δt and Δm into 50 and 200 bins respectively.

		Parameters
		----------
		uids : array_like, list of qso uids whose lightcurves are to be used.
		n_bins_t  : int, total number of time bins.
		n_bins_m  : int, number of mag bins.
		t_max	 : float, set this to largest ∆t in self.df (7600 for all surveys in obs frame, 3010 for all surveys in rest frame).
		t_spacing : str, ('log' or 'lin') for logarithmic and linear bin spacing respectively.
		m_spacing : str, ('log' or 'lin') for logarithmic and linear bin spacing respectively.
		read_in   : boolean, True to read in from disk, False to compute and return.
		key	   : str, property from VAC.
		ztf	   : boolean, True to use ztf data only.

		Returns
		-------
		dms_binned  : array_like, (19, n_bins_m)
		dts_binned  : array_like, (19, n_bins_t)
		t_bin_edges : array_like, (n_bins_t+1, )
		t_bin_chunk : array_like, ()
		t_bin_chunk_centres : array_like, (), centres of t bin chunk (NOT geometric average, could try this)
		m_bin_edges : array_like, ()
		m_bin_centres : array_like, ()
		t_dict	  : dict,
		"""
		if m_spacing == 'log':
			def calc_m_edges(n_bins_m, steepness):
				start = np.log10(steepness)
				stop = np.log10(steepness+3)
				return np.concatenate((-np.logspace(start,stop,int(n_bins_m/2+1))[:0:-1]+steepness,np.logspace(start,stop,int(n_bins_m/2+1))-steepness))
			m_bin_edges = calc_m_edges(200,0.2)
		elif m_spacing == 'lin':
			m_bin_edges = np.linspace(-3,3,201)

		if t_spacing == 'log':
			def calc_t_bins(t_max, n_bins_t=19 , steepness=10):
				start = np.log10(steepness)
				stop = np.log10(steepness+t_max)
				return np.logspace(start,stop,n_bins_t+1)-steepness
			t_bin_chunk = calc_t_bins(t_max = t_max, steepness = 1000)
		elif t_spacing == 'lin':
			t_bin_chunk = np.linspace(0,t_max,20)

		t_bin_edges = np.linspace(0,t_max,(n_bins_t+1))
		t_dict = dict(enumerate(['{0:1.0f}<t<{1:1.0f}'.format(t_bin_chunk[i],t_bin_chunk[i+1]) for i in range(len(t_bin_chunk)-1)]))
		dts_binned = np.zeros((19,n_bins_t))
		dms_binned = np.zeros((19,n_bins_m))
		m_bin_centres = (m_bin_edges[1:] + m_bin_edges[:-1])/2
		t_bin_chunk_centres = (t_bin_chunk[1:] + t_bin_chunk[:-1])/2

# 		dms_binned = np.loadtxt('analysis/qsos/computed/dtdm/{}/dms_binned_{}_{}.csv'.format(key,key,read_in), delimiter = ',')
# 		dts_binned = np.loadtxt('analysis/qsos/computed/dtdm/{}/dms_binned_{}_{}.csv'.format(key,key,read_in), delimiter = ',')
		dtdms = [np.empty((0,2))]*19
		idxs = np.digitize(dtdm[:,0], t_bin_chunk)-1
		for index in np.unique(idxs): #Can we vectorize this?
			dtdms[index] = np.append(dtdms[index],dtdm[(idxs == index),:],axis = 0)

		print('now binning')
		for i in range(19):
			print('dtdm counts in {}: {:,}'.format(t_dict[i],len(dtdms[i])))
			dts_binned[i] += np.histogram(dtdms[i][:,0], t_bin_edges)[0]
			dms_binned[i] += np.histogram(dtdms[i][:,1], m_bin_edges)[0]

		return dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict

	def save_dtdm_rf(self, uids, time_key):

		sub_df = self.df[[time_key, 'mag', 'catalogue']].loc[uids]

		df = pd.DataFrame(columns=[self.ID, 'dt', 'dm', 'cat'])
		for uid in uids:
			#maybe groupby then iterrows? faster?
			group   = sub_df.loc[uid]
			mjd_mag = group[[time_key,'mag']].values
			cat	 = group['catalogue'].values
			n = len(mjd_mag)
			# dtdm defined as: ∆m = (m2 - m1), ∆t = (t2 - t1) where (t1, m1) is the first obs and (t2, m2) is the second obs.
			# Thus a negative ∆m corresponds to a brightening of the object
			dcat = 3*cat + cat[:,np.newaxis]
			dcat = dcat[np.triu_indices(n,1)]

			dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
			dtdm = dtdm[np.triu_indices(n,1)]
			dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

			duid = np.full(int(n*(n-1)/2),uid,dtype='uint32')
			# collate data to DataFrame and append
			df = df.append(pd.DataFrame(data={self.ID:duid,'dt':dtdm[:,0],'dm':dtdm[:,1],'cat':dcat}))
			
			if (uid % 500 == 0):
				print(uid)

		return df

	def save_error(self, uids):

		sub_df = self.df[['magerr']].loc[uids]

		df = pd.DataFrame(columns=[self.ID, 'magerr'])
		for uid in uids:
			#maybe groupby then iterrows? faster?
			group   = sub_df.loc[uid]
			magerr	= group['magerr'].values
			n       = len(magerr)			

			# dtdm defined as: ∆m = (m2 - m1), ∆t = (t2 - t1) where (t1, m1) is the first obs and (t2, m2) is the second obs.
			# Thus a negative ∆m corresponds to a brightening of the object
			dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
			dmagerr = dmagerr[np.triu_indices(n,1)]

			uid = np.full(n,uid,dtype='uint32')

			# collate data to DataFrame and append
			df = df.append(pd.DataFrame(data={self.ID:uid,'dmagerr':dmagerr}))
			
			if (uid % 500 == 0):
				print(uid)

		return df

	def plot_sf_moments_pm(self, key, bounds, save = False, t_max=3011, ztf=False):
		"""
		Plots both positive and negative structure functions to investigate asymmetry in light curve.

		Parameters
		----------
		key : str
				Property from VAC
		bounds : array of z scores to use


		Returns
		-------
		fig, ax, fig2, axes2, fig3, axes3: axes handles
		"""
		fig, ax = plt.subplots(1,1,figsize = (16,8))
		fig2, axes2 = plt.subplots(2,1,figsize=(16,10))
		fig3, axes3 = plt.subplots(8,1,figsize = (16,50))
		label_range = {i:'{:.1f} < z < {:.1f}'.format(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)}
		label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}
#				label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
		label_moment = ['mean', 'Excess kurtosis']
		cmap = plt.cm.jet
		for i in range(8):
			dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = t_max, t_spacing='log', m_spacing='log', read_in=i+1, key=key, ztf=ztf)
			SF_n = (((m_bin_centres[:100]**2)*dms_binned[:,:100]).sum(axis=1)/dms_binned[:,:100].sum(axis=1))**0.5
			SF_p = (((m_bin_centres[100:]**2)*dms_binned[:,100:]).sum(axis=1)/dms_binned[:,100:].sum(axis=1))**0.5
			ax.plot(t_bin_chunk_centres, SF_p, label = label_range_val[i], lw = 0.5, marker = 'o', ls='-',  color = cmap(i/10))
			ax.plot(t_bin_chunk_centres, SF_n, label = label_range_val[i], lw = 0.5, marker = 'o', ls='--', color = cmap(i/10))
			ax.legend()
			ax.set(yscale='log', xscale='log')

			axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,:100].sum(axis=1), alpha = 0.5, label = '-ve',bins = 19)
			axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,100:].sum(axis=1), alpha = 0.5, label = '+ve',bins = 19)
			axes3[i].set(yscale='log')
			dms_binned_norm = np.zeros((19,200))
			moments = np.zeros(19)
			for j in range(19):
				dms_binned_norm[j],_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[j], density=True);
#								print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
#								print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
			moments = calc_moments(m_bin_centres,dms_binned_norm)


			for idx, ax2 in enumerate(axes2.ravel()):
				ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
		#				ax2.legend()
				ax2.set(xlabel='mjd', ylabel = label_moment[idx])
				ax2.axhline(y=0, lw=0.5, ls = '--')

#								ax2.title.set_text(label_moment[idx])
		ax.set(xlabel='mjd', ylabel = 'structure function')
		if save:
			# fig.savefig('SF_{}.pdf'.format(key),bbox_inches='tight')
			fig2.savefig('moments_{}.pdf'.format(key),bbox_inches='tight')

		return fig, ax, fig2, axes2, fig3, axes3

	def plot_sf_moments(self, key, bounds, save = False, t_max=3011, ztf=False):
		fig, ax = plt.subplots(1,1,figsize = (16,8))
		fig2, axes2 = plt.subplots(2,1,figsize=(16,10))
		fig3, axes3 = plt.subplots(8,1,figsize = (16,50))
		label_range = {i:'{:.1f} < z < {:.1f}'.format(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)}
		label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}
#				label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
		label_moment = ['mean', 'Excess kurtosis']
		cmap = plt.cm.jet
		for i in range(8):
			dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = t_max, t_spacing='log', m_spacing='log', read_in=i+1, key=key, ztf=ztf)
			SF = (((m_bin_centres**2)*dms_binned).sum(axis=1)/dms_binned.sum(axis=1))**0.5
			ax.plot(t_bin_chunk_centres, SF, label = label_range_val[i], lw = 0.5, marker = 'o', ls='-',  color = cmap(i/10))
			ax.legend()
			ax.set(yscale='log', xscale='log')

			axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned.sum(axis=1), alpha = 0.5,bins = 19)
			axes3[i].set(yscale='log')
			dms_binned_norm = np.zeros((19,200))
			moments = np.zeros(19)
			for j in range(19):
				dms_binned_norm[j],_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[j], density=True);
#								print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
#								print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
			moments = calc_moments(m_bin_centres,dms_binned_norm)


			for idx, ax2 in enumerate(axes2.ravel()):
				ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
		#				ax2.legend()
				ax2.set(xlabel='mjd', ylabel = label_moment[idx])
				ax2.axhline(y=0, lw=0.5, ls = '--')

#								ax2.title.set_text(label_moment[idx])
		ax.set(xlabel='mjd', ylabel = 'structure function')
		if save == True:
			# fig.savefig('SF_{}.pdf'.format(key),bbox_inches='tight')
			fig2.savefig('moments_{}.pdf'.format(key),bbox_inches='tight')

		return fig, ax, fig2, axes2, fig3, axes3

	def plot_sf_ensemble(self, save = False):
		fig, ax = plt.subplots(1,1,figsize = (16,8))
		dms_binned_tot = np.zeros((8,19,200))
		for i in range(8):
			dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = 3011, t_spacing='log', m_spacing='log', read_in=i+1, key=key)
			dms_binned_tot[i] = dms_binned

		dms_binned_tot = dms_binned_tot.sum(axis=0)

		SF = (((m_bin_centres**2)*dms_binned_tot).sum(axis=1)/dms_binned_tot.sum(axis=1))**0.5
		ax.plot(t_bin_chunk_centres,SF, lw = 0.5, marker = 'o')
		ax.set(yscale='log',xscale='log')
		ax.set(xlabel='mjd',ylabel = 'structure function')
		if save == True:
			fig.savefig('SF_ensemble.pdf',bbox_inches='tight')

	def plot_series(self, uids, survey=None):
		"""
		Plot lightcurve of given objects

		Parameters
		----------
		uids : array_like
				uids of objects to plot
		catalogue : int
				Only plot data from given survey
		survey : 1 = SDSS, 2 = PS, 3 = ZTF

		"""
		fig, axes = plt.subplots(len(uids),1,figsize = (20,3*len(uids)), sharex=True)
		if len(uids)==1:
			axes=[axes]
			
		for uid, ax in zip(uids,axes):
			single_obj = self.df.loc[uid].sort_values('mjd')
			for band in 'ugriz':
				single_band = single_obj[single_obj['filtercode']==band]
				if survey is not None:
					single_band = single_band[single_band['catalogue']==survey]
				for cat in single_band['catalogue'].unique():
					x = single_band[single_band['catalogue']==cat]
					ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 0.2, markersize = 3, marker = self.marker_dict[cat], label = self.survey_dict[cat]+' '+band, color = self.plt_color[band])
			ax.invert_yaxis()
			ax.set(xlabel='mjd', ylabel='mag')
	# 			ax.legend(loc=)
			ax.text(0.02, 0.9, 'uid: {}'.format(uid), transform=ax.transAxes, fontsize=10)
		
		plt.subplots_adjust(hspace=0)
		
		return fig, ax
			
	def plot_series_bokeh(self, uids, survey=None):
		"""
		Plot lightcurve of given objects using bokeh

		Parameters
		----------
		uids : array_like
				uids of objects to plot
		catalogue : int
				Only plot data from given survey
		survey : 1 = SDSS, 2 = PS, 3 = ZTF

		"""

		plots = []
		for uid in uids:
			single_obj = self.df.loc[uid]
			if survey is not None:
				single_obj = single_obj[single_obj['catalogue']==survey]
			p = figure(title='uid: {}'.format(uid), x_axis_label='mjd', y_axis_label='r mag', plot_width=1000, plot_height=400)
			for cat in single_obj['catalogue'].unique():
				mjd, mag = single_obj[single_obj['catalogue']==cat][['mjd','mag']].sort_values('mjd').values.T
				p.scatter(x=mjd, y=mag, legend_label=self.survey_dict[cat], marker=self.marker_dict_bokeh[cat], color=self.plt_color_bokeh)
				p.line   (x=mjd, y=mag, line_width=0.5, color=self.plt_color_bokeh)
			p.y_range.flipped = True
			plots.append(p)

		show(column(plots))


	def plot_property_distributions(self, prop_range_dict, n_width, n_bins = 250, separate_catalogues = True):
		"""
		Parameters
		----------
		prop_range_dict : dict
				dictionary of keys and their ranges to be used in histogram
		n_width : int
				width of subplot
		n_bins : int
				number of bins to use in histogram
		separate_catalogues : boolean
				if True, plot data from each survey separately.
		"""
		m = -( -len(prop_range_dict) // n_width )
		fig, axes = plt.subplots(m, n_width,  figsize = (5*n_width,5*m))
		cat_label_dict = {1:'SDSS', 2:'PanSTARRS', 3:'ZTF'}
		for property_name, ax, in zip(prop_range_dict, axes.ravel()):
			if separate_catalogues == True:
				for cat, color in zip(self.cat_list,'krb'):
					self.properties[self.properties['catalogue']==cat][property_name].hist(bins = n_bins, ax=ax, alpha = 0.3, color = color, label = cat_label_dict[cat], range=prop_range_dict[property_name]);
				ax.legend()
			elif separate_catalogues == False:
				self.properties[property_name].hist(bins = 250, ax=ax, alpha = 0.3, range=prop_range_dict[property_name]);
			else:
				print('Error, seperate_catalogues must be boolean')
			ax.set(title = property_name)
