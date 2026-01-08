# DicomSet

# Tenets

## Referencing objects by ID.
Don't make people write 'pat_id' etc. in the API, it's kind of redundant.

How do we refer to patients/studies/series/regions/landmarks/etc. in the code? Do we use pat_id or pat, region or region?
'_id' seems a bit pointless, also why don't we say 'dataset_id'. Maybe just set.list_patients(pat='PMCC_ReIrrad_L01') for example.
Or plot_patient('PMCC-REIRRAD', 'PMCC_ReIrrad_L01', landmark='all').

For instances where a function API takes data as well (e.g. plot_image takes landmarks (IDs), plus landmarks data) the data should be referenced explicitly, (landmarks_data=), whilst the landmark IDs are implicity (landmark=). This is primarily because we don't often pass data around but we always pass around IDs.

## Referencing regions/landmarks.
As method args: Do we use singular or plural? In cases where both singular/plural objects are accepted, e.g. plot_patient(dataset, pat_id) accepts single/multiple patients, we use the singular. Use the plural only in cases where the singular is not accepted, i.e. only List[RegionID] is supported.

As method names and variable names:
When referring to regions/landmarks series or data this is a special case as the series contains multiple regions or landmarks, whereas a CT series will only contain one CT image. Therefore, we use something like 'study.regions/landmarks_series()' and 'study.regions/landmarks_data()'. For 'has_landmarks' - this could refer to the presence of a landmarks series, whist 'has_landmark' refers to the presence of a particular landmark ID.

# Plotting

## How do we deal with expansion of patients/studies and series?

- Patients are expanded down the rows - easy!
- Studies and series are expanded along the columns, grouped by studies.

How do we add 'landmark_series'? What's our current use case?
I have 3 studies (moving, fixed, fixed) and I'd like to show different landmark series for each of them.
So passing landmark_series=[series_1, series_1, series_2] should choose series_2 for the second fixed image.
When we pass 'series' it refers to the image series for each study.
- One study, multiple series IDs, the series IDs are assumed to belong to the single study.
- Multiple studies, one series ID, the series is assumed to belong to each study.
- Multiple studies, multiple series IDs, the the number of studies and series IDs should match.
We want the same behaviour for 'landmark_series'.
- One study, multiple image series, landmark series are split over image series.
- Multiple studies, one image series, landmark series are split over image series of different studies.
- Multiple... landmark series are split over

## Should we allow plotting of predictions?

- This could perhaps be a separate codebase.
- We also should just allow propagation of predictions to the main codebase. These would become new series
that can be referenced in the normal patient plotting code.

## Do we need to reference series IDs for registration?

- Currently the file structure only goes down to study IDs, but we could be registering different series within
the same study. Very common, CT and MR registration.
- Should we provide series ID shortcuts? This is very useful for any API that we'll call regularly, e.g. loading registration data,
or plotting, but not necessary for the depths of the code base - just pass everything as args not kwargs.

## Small ideas

- There's a definite difference between what you call a variable in a method argument and what might be best in the code body. For example,
you might want to shorten something for the API for ease of use (e.g. pat, study), whereas pat_id/study_id allows you to differentiate 
between IDs and objects in the code base. Same thing for pat in the API and fixed_pat in the codebase.
