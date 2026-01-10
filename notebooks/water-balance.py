# %% [markdown]
#  # Applying the water balance equation
# 
# 
# 
#  ## Objectives
# 
#  - Apply the water balance equation to real-world data
# 
#  - Test the impact of land cover on the runoff ratio
# 
#  - Fit a simple statistical model to data
# 
# 
# 
#  ## Prerequisites:
# 
#  - Basic understanding of Python
# 
#  - Familiarity with Pandas, Matplotlib

# %% [markdown]
#  ## Dataset
# 
#  We will be using the CAMELS-GB dataset. This contains daily hydrometeorological data for around 670 catchments in Great Britain, as well as catchment attributes related to land use/land cover, geology, and climate. Download the data [here](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9), then upload it to the (currently) empty `data` folder in this repository. Now extract the data archive: 

# %%
import os
import zipfile

zip_path = 'data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip'
extract_dir = 'data/8344e4f3-d2ea-44f5-8afa-86d2987543a9'

if not os.path.isdir(extract_dir):
    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall('data')
    except FileNotFoundError:
        print(f"Zip file {zip_path} not found.")
    except zipfile.BadZipFile:
        print(f"Zip file {zip_path} is corrupt.")

# %% [markdown]
# Let's create a path variable so that we can easily navigate to the data files:

# %%
import os
DATADIR = os.path.join('data', '8344e4f3-d2ea-44f5-8afa-86d2987543a9', 'data')


# %% [markdown]
#  Now Load the data for a catchment chosen at random. The timeseries data are stored as csv files, so we use Pandas to load them into a Pandas DataFrame object:

# %%
import pandas as pd
id = '97002'
data = pd.read_csv(os.path.join(DATADIR, 'timeseries', f'CAMELS_GB_hydromet_timeseries_{id}_19701001-20150930.csv'), parse_dates=[0])
data.head()

# %% [markdown]
# Look at the CAMELS-GB [manuscript](https://doi.org/10.5194/essd-12-2459-2020) and find out the units of each variable. Verify that `discharge_spec` is consistent with `discharge_vol` (HINT: you will need to find the drainage area of your chosen catchment so you can convert the volume to a depth. For now, you can find out this information by looking at the [NRFA website](https://nrfa.ceh.ac.uk/data/search). Later we will use the static catchment attributes provided with CAMELS-GB). 

# %% 
raise NotImplementedError()

# %% [markdown]
#  Later on it will be useful to have the catchment ID in the dataframe, so we add it here:

# %%
data['id'] = id

# %% [markdown]
# We can also see that the `discharge_vol` column contains `NaN` values - these usually indicate missing data. There are various things we can do to handle (or impute) missing values, but for now let's just remove them: 

# %%
data = data.dropna(subset=['discharge_vol'])
data.head()

# %% [markdown]
#  Recall the water balance equation from lecture 1:
# 
#  $\frac{dS}{dt} = P - E - Q$
# 
#  where $\frac{dS}{dt}$ is the change in storage over time, $P$ is precipitation, $E$ is evaporation and $Q$ is streamflow. Also recall that over long time periods we can assume the storage term tends towards zero. Now we can write:
# 
#  $0 = P - E - Q$
# 
#  and hence:
# 
#  $E = P - Q.$
# 
#  This is convenient because evaporation is hard to measure accurately. Let's use the equation above to estimate the catchment-averaged evaporation. We will work at annual timescales so that we can reasonably assume that the storage term is negligible. First we need to compute the annual precipitation and discharge. To do this we typically use the "water year" instead of the calendar year. This avoids the potential for large errors in the water balance because catchment storage can vary significantly during the wet season. In the UK the water year is taken as 1st October to 30th September. Fortunately Pandas has some magic that allows us to easily aggregate by water year:

# %%
data['water_year'] = data['date'].dt.to_period('Y-SEP')
data.head()


# %% [markdown]
#  Here, `A-SEP` is a period alias for "annual frequency, anchored end of September". Learn more about period aliases by consulting the [Pandas documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-period-aliases).
# 
# 
# 
#  We also need to convert the discharge from m3/s to m3/day:

# %% [markdown]
#  Now we group our dataframe by the new `water_year` column, and compute the sum of precipitation and discharge. Before doing this we need to convert discharge_vol from m3/s (daily average discharge) to m3/day (daily total discharge):

# %%
data['discharge_vol'] *= 60 * 60 * 24 # m3/s -> m3/day
anndata = data.groupby(['id', 'water_year'])[['precipitation', 'pet', 'discharge_spec', 'discharge_vol']].sum().reset_index()


# %% [markdown]
#  Aggregating data is an extremely useful skill in hydrology. Think about how you might use Pandas to aggregate by month or by season.
# 
# 
# 
#  When making comparisons between catchments, it is common to transform all variables to a *depth* so that the effect of catchment area is reduced. This allows us to compare the hydrological behaviour of a large catchment (e.g. Tweed) with a much smaller catchment. Let's load the catchment attributes and find the area of our catchment.

# %%
metadata = pd.read_csv(os.path.join(DATADIR, f'CAMELS_GB_topographic_attributes.csv'))
metadata['gauge_id'] = metadata['gauge_id'].astype(str)
area = metadata[metadata['gauge_id'] == id]['area'].values[0]
area *= 1e6 # km2 -> m2 


# %% [markdown]
#  Let's return to the question I posed above, about verifying that discharge_spec is consistent with discharge_vol. Let's transform our volumetric data to depth units:

# %%
anndata['discharge_spec_computed'] = anndata['discharge_vol'].copy() # m3/day
anndata['discharge_spec_computed'] /= area # m3 -> m
anndata['discharge_spec_computed'] *= 1000 # m -> mm


# %% [markdown]
#  If you look at the dataframe, you see that column `discharge_vol` is now the same as `discharge_spec`. In future, you can use `discharge_spec` directly, without the need for transformation. We now have everything we need to estimate evaporation using the water balance equation:

# %% 
anndata['diff'] = anndata['discharge_spec'] - anndata['discharge_spec_computed']
print(anndata[['discharge_spec', 'discharge_spec_computed', 'diff']].head())
print(anndata['diff'].abs().mean())

# %%
anndata['evaporation'] = anndata['precipitation'] - anndata['discharge_spec_computed']


# %% [markdown]
#  Let's plot this data:

# %%
import matplotlib.pyplot as plt

anndata = anndata.set_index('water_year')
anndata.plot(y=['precipitation', 'discharge_spec', 'evaporation'], figsize=(12, 6))

plt.title(f'Water balance for catchment {id}')
plt.xlabel('Water year end')
plt.ylabel('Depth (mm)')
plt.legend(['Precipitation', 'Discharge', 'Evaporation'])
plt.grid(True)
plt.show()

anndata = anndata.reset_index()

# %% [markdown] 
# Instead of estimating evaporation at the annual scale, we could also do this at the monthly or seasonal scale. Here is a helpful code snippet for adding the season to the dataframe:

# %% 
def month_to_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    else:
        return 'SON'

# Align year to start in September - this ensures that DJF (which spans two calendar years) is grouped correctly
data['season_year'] = data['date'].dt.to_period('Y-AUG')
data['season'] = data.date.dt.month.apply(month_to_season)
seasondata = data.groupby(['id', 'season_year', 'season'])[['precipitation', 'pet', 'discharge_spec', 'discharge_vol']].sum().reset_index()

# %% [markdown]
# Now you can compute seasonal evaporation in the same way as before. What do you notice about the seasonal evaporation? Are there any seasons where evaporation is negative? What might this imply? 

# %% 
raise NotImplementedError()

# %% [markdown]
#  ## Land cover impacts
# 
#  We will cover the drivers of evaporation in more detail later on the course. One question we may have is the role of different land cover types on the water balance. Let's investigate whether land use impacts evaporation by looking at some forested catchments:

# %%
metadata_lu = pd.read_csv(os.path.join(DATADIR, f'CAMELS_GB_landcover_attributes.csv'))


# %% [markdown]
#  Have a look at the columns in `metadata_lu` and consult Coxon et al. (2020). Which columns represent forest? Create a new column called `forest_perc` that combines the two types.

# %%
raise NotImplementedError()


# %% [markdown]
#  To compare the impact of vegetation on runoff generation, it would be useful to compute a summary measure for each catchment. One such measure, or signature, is the runoff ratio, defined as the proportion of precipitation that becomes runoff. We can calculate this as follows:

# %%
anndata_sum = anndata.groupby('id')[['precipitation', 'discharge_spec_computed']].sum()


# %% [markdown]
#  Now we can calculate the runoff ratio:

# %%
anndata_sum['runoff_ratio'] = anndata_sum['discharge_spec_computed'] / anndata_sum['precipitation']

# %% [markdown]
# What does the runoff ratio tell us about the catchment? Would you expect the number to be higher or lower for an arid catchment? What about a humid catchment? 

# %% [markdown]
# ## Optional exercise
# 
# 1. Divide the catchments into three groups: Low forest (<10%), Medium forest (10-30%) and High forest (>30%).
# 2. Compute the runoff ratio for every catchment (HIND: write a loop to perform the steps outlined above). 
# 3. Make a boxplot showing the distribution of runoff ratios for each group. 
# 4. Can you see a clear pattern? Consider the following questions: 
#    (i)   Is this is a fair comparison (HINT: look at the number of catchments in each group)?
#    (ii)  What other factors might be influencing the runoff ratio? 
#    (iii) How could you improve this analysis? How might statistical or machine learning models help?
raise NotImplementedError()

