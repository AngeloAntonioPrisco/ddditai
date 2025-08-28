# Data Understanding – Dddit AI

<p align="center"><img src='https://i.postimg.cc/SNSGrSv2/dddit-ai-upscaled.png' alt="Quixel_Texel_Logo" height="400"></p>

## 1. Initial Strategy

**Section Description:**  
This section describes the initial plan for collecting and understanding the data. It represents the intended strategy before actual data retrieval, including what features were expected to be available, the sources, and the methodology.

**Objective:**  
Automatically collect an initial dataset of 500 FBX models from Sketchfab, extracting all relevant metadata and user-defined tags for exploratory analysis and model training.

**Planned Data Sources:**  
- Sketchfab API v3 (`/v3/models` and `/v3/models/{uid}`)  
- Retrieval automated (no manual UID insertion)  
- Multithreading to speed up collection  

**Metadata to Extract:**  

<div style="display: table; margin: auto;">

| Feature               | Description                                                                |
|-----------------------|----------------------------------------------------------------------------|
| triangles             | Number of triangles in the mesh (measure of model complexity)              |
| vertices              | Number of vertices in the mesh                                             |
| materials             | Number of materials used in the model                                      |
| textures              | Number of textures applied                                                 |
| UV layers             | Presence and number of UV mapping layers                                   |
| vertex colors         | Whether the model contains vertex color information                        |
| animations            | Number of animations present in the model                                  |
| rigged geometries     | Whether the model is rigged for animation                                  |
| morph geometries      | Number of morph targets for shape changes                                  |
| scale transformations | Whether the model has applied scale transformations                        |
| user tags             | Tags provided by the model creator                                         |
| user categories       | Category selected by the user (e.g., prop, character, environment, weapon) |

</div>

**Storage Plan:**  
- Data to be logged and versioned in MLflow  
- Exportable to CSV for offline analysis

**Note:**  
It was decided **not** to download the FBX models due to time and computational restrictions. Downloading models would require significantly more processing time and RAM. Instead, the system relies solely on metadata accessible via the API, which allows fast multithreaded retrieval.

## 2. Findings / Discoveries

**Section Description:**  
This section tracks discoveries about data retrieving process.
It highlights all information that has been gathered based on subsequent
studies and that was not taken into account when planning the original strategic plan.

**Objective:**  
Take decisions and finding solutions to any deviations from the original strategic plan.

### Unavailable data
During the study of Sketchfab API turned out that `/v3/models/{uid}` API don't permit to gather all features described
in the original strategic plan. In particular:

<div style="display: table; margin: auto;">

| Feature               | Retrievable via API? | Notes / Alternatives                |
|-----------------------|----------------------|-------------------------------------|
| vertices              | ✅                    |                                     |
| materials             | ✅                    |                                     |
| textures              | ✅                    |                                     |
| animations            | ✅                    |                                     |
| user tags             | ✅                    |                                     |
| user categories       | ✅                    |                                     |
| triangles             | ❌                    | Web scraping or GLTF temporary load |
| UV layers             | ❌                    | Web scraping or GLTF temporary load |
| vertex colors         | ❌                    | Web scraping or GLTF temporary load |
| rigged geometries     | ❌                    | Web scraping or GLTF temporary load |
| morph geometries      | ❌                    | Web scraping or GLTF temporary load |
| scale transformations | ❌                    | Web scraping or GLTF temporary load |

</div>

An alternative method for retrieving missing features can be: 
- **Web Scraping:** parse HTML of model pages to extract features not exposed by the API  
- **Temporary GLTF Download:** load models in RAM only for feature extraction, using multithreading

### Unavailable temporary GLTF download
During the study of Sketchfab download API turned out that `/v3/models/{uid}/download` API requires
an additional authorization token in addition to the one obtained when registering a Sketchfab account.
To obtain the token that enables downloading, you must contact Sketchfab and initiate
a procedure at the end of which the token is transmitted.
Due to time constraints, the current solution is to exclude solutions that involve
downloading models in GLTF format and find a solution based on Web Scraping.

### Available data using Sketchfab API
Further investigation of Sketchfab `/v3/models/{uid}` API has allowed to retrieve
all the features that can be retrieved from API itself. In particular:

<div style="display: table; margin: auto;">

| Feature                | JSON Source         | Notes                                                    |
|------------------------|---------------------|----------------------------------------------------------|
| UID                    | uid                 | Unique identifier of the model                           |
| Model Name             | name                | Textual name                                             |
| Published Date         | publishedAt         | ISO date                                                 |
| Updated Date           | updatedAt           | ISO date                                                 |
| Like Count             | likeCount           | Number of likes                                          |
| Comment Count          | commentCount        | Number of comments                                       |
| Is Downloadable        | isDownloadable      | Boolean, indicates if download is theoretically possible |
| Is Age Restricted      | isAgeRestricted     | Boolean                                                  |
| PBR Type               | pbrType             | Declared PBR type (can be empty string)                  |
| Material Count         | materialCount       | Number of declared materials                             |
| Texture Count          | textureCount        | Number of declared textures                              |
| Vertex Count           | vertexCount         | Number of declared vertices (not triangles)              |
| Animation Count        | animationCount      | Number of declared animations                            |
| Sound Count            | soundCount          | Number of audio tracks                                   |
| Face Count             | faceCount           | Number of faces, if available                            |
| Tags                   | tags                | List of tags (slug or URI)                               |
| User Info              | user                | Username, displayName, avatar, profile                   |
| Categories             | categories          | Name, slug, UID                                          |
| License                | license             | License type                                             |
| Views                  | viewCount           | Number of views                                          |
| Thumbnails             | thumbnails          | List of preview image URLs                               |
| Embed URL / Viewer URL | embedUrl, viewerUrl | Direct link to the model embed/viewer                    |

</div>

This approach can help define a solution without adding extra complexity 
(e.g., GLTF downloads or web scraping) while partially recovering some of
the features identified in the original strategic plan. In particular:

<div style="display: table; margin: auto;">

| Old Feature           | New Feature       | Notes                                                                                                                                                                        |
|-----------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -                     | uid               | Newly considered feature. Unique identifier of the model. Not planned initially.                                                                                             |
| -                     | is age restricted | Newly considered feature. Boolean indicating if the model is age restricted. Not part of original planning.                                                                  |
| -                     | pbr type          | Newly considered feature. Declared PBR type (can be empty string, providing no information). Not planned initially.                                                          |
| -                     | texture count     | Newly considered feature. Number of declared textures. Not part of original planning.                                                                                        |
| triangles             | face count        | Face count can approximate triangles since usually one face corresponds to two triangles in most workflows.                                                                  |
| UV layers             | material count    | Material count can act as a substitute for UV layers. Originally UV layers were considered a measure of model texturing complexity; material count serves a similar purpose. |
</div>

The following recap consolidates the original strategic plan together with the new findings:

<div style="display: table; margin: auto;">

| Feature               | Retrievable via API? | Notes / Alternatives                                        |
|-----------------------|----------------------|-------------------------------------------------------------|
| uid                   | ✅                    | Newly added feature. Unique identifier of the model.        |
| is age restricted     | ✅                    | Newly added feature. Boolean indicating age restriction.    |
| pbr type              | ✅                    | Newly added feature. Can be empty string if not defined.    |
| texture count         | ✅                    | Newly added feature. Number of declared textures.           |
| vertices              | ✅                    |                                                             |
| materials             | ✅                    |                                                             |
| animations            | ✅                    |                                                             |
| user tags             | ✅                    |                                                             |
| user categories       | ✅                    |                                                             |
| face count            | ✅                    | Substitutes triangles (approximation: 1 face ≈ 2 triangles) |
| material count        | ✅                    | Substitutes UV layers (measure of texturing complexity)     |
| vertex colors         | ❌                    | Not substituted, will not be considered for now             |
| rigged geometries     | ❌                    | Not substituted, will not be considered for now             |
| morph geometries      | ❌                    | Not substituted, will not be considered for now             |
| scale transformations | ❌                    | Not substituted, will not be considered for now             |

</div>

### Limits and quotas

From usage of Sketchfab `/v3/models` API emerged that calls to the API can be throttled to limit abuse.
When an application is being throttled, it will receive a `429 Too Many Requests response`.
This means that workflow result interrupted and must wait before making more requests.
This implies updating the process to implement a backoff strategy that retries
requests using an exponential backoff, up to a maximum delay of 1 hour.

From further studies on these limits, it emerged that Sketchfab does not use any 
headers to indicate a recommended retry time.
In other, the limits do not seem to be related to IP addresses but rather to the token.
To avoid such issues, a possible solution is to use multiple tokens, one per thread, and longer
pause periods.
In particular, it seems reasonable to wait a fixed time, such as 90 seconds,
after a batch of requests (e.g., 32 requests), and to pause for one or two seconds between
consecutive requests. Using four threads and 256 models per tag with 8 tags,
this process would take approximately 10 minutes.

## 3. Final Strategy

**Section Description:**
This section outlines the final plan for collecting and understanding the data.
It represents the intended strategy before performing actual data retrieval,
taking into account all insights documented in the Findings / Discoveries section.
If new findings emerge, this section may be updated accordingly. 
Each update should be committed (push) to maintain a clear history of the evolution 
of the chosen methodology throughout the data retrieval process.

**Objective:**
Define the final methodology and describe the dataset to be retrieved, consolidating the approach that will guide the data collection process.

**Planned Data Sources:**  
- Sketchfab API v3 (`/v3/models` and `/v3/models/{uid}`)  
- Retrieval automated (no manual UID insertion)  
- Multithreading is used to accelerate data collection,
with pause intervals and an exponential backoff algorithm to avoid hitting request limits.
Each thread is associated with a list of tags it is responsible for retrieving, as well as a distinct token for making requests, 
- List of tag to perform an informed data retrieving

**Metadata to Extract:**  

<div style="display: table; margin: auto;">

| Feature           | Description                                                             |
|-------------------|-------------------------------------------------------------------------|
| uid               | Unique identifier of the model.                                         |
| associated_tag    | The tag assigned to the thread responsible for retrieving related data. |
| is age restricted | Boolean indicating if the model has age restriction.                    |
| pbr type          | Declared PBR type, can be an empty string if not defined.               |
| texture count     | Number of declared textures.                                            |
| vertices          | Number of declared vertices of the model.                               |
| animations        | Number of declared animations associated with the model.                |
| user tags         | List of user-defined tags attached to the model.                        |
| user categories   | Categories assigned to the model.                                       |
| face count        | Number of declared faces. Considered a substitute for triangles.        |
| material count    | Number of declared materials in the model.                              |

</div>

**Storage Plan:**  
- Data to be logged and versioned in MLflow  
- Exportable to CSV for offline analysis
- Credits to models' author to be logged and versioned in MLflow  

## 4. Data Exploration

**Section Description:**  
This section describes the initial exploration of the dataset retrieved during the data understanding phase.  
It highlights patterns, distributions, and potential correlations in the data, with the goal of identifying anomalies, missing values, or relationships that can guide preprocessing and imputation strategies.
If new findings emerge, this section may be updated accordingly. 
Each update should be committed (push) to maintain a clear history of the evolution 
of the chosen studies to perform.

**Objective:**  
- Compute standard descriptive statistics for numerical features (mean, median, std, min, max, quartiles).
- Visualize box plots for each feature and using them to define how many outliers characterize the dataset.
- Visualize distributions of numerical and boolean features using histograms.  
- Analyze correlations between all features in relation to `associated_tag` feature. 
- Analyze correlations between all following pair of features: 
  - `material count` vs `pbr type`  
  - `texture count` vs `pbr type`
- Detect and quantify extreme values, e.g., number of models with `face count` > 200,000.
This is necessary because many CC0 models are created via photogrammetry, resulting in very high face counts that are not representative of the target models expected to be pushed to Dddit.
- Provide insights for handling missing or empty fields (e.g., empty `pbr type`) and guide potential data imputation based on correlated features.

## 5. Result

**Section Description:**  
This section describes the initial exploration of the dataset retrieved during the data understanding phase.  
It highlights patterns, distributions, and potential correlations in the data, with the goal of identifying anomalies, missing values, or relationships that can guide preprocessing and imputation strategies.
If new results are obtained, this section may be updated accordingly. 
Each update should be committed (push) to maintain a clear history of the results obtained
in relation of the methodologies and studies defined in other sections.

**Objective:**
Track results of data exploration process and take notes of relevant information
to improve data exploration process.

The data analysis of the MLflow run `ca8067b0d2bb4cbdb7a947090eb377fe` reveals several key insights regarding feature distributions, correlations, and data quality.
The features `uid`, `pbr_type`, `texture_count`, `vertex_count`, `material_count`, `animation_count`, `user_tags`, `user_categories`,
and `face_count` all show statistically significant associations with `associated_tag`, while `is_age_restricted` does not.
These significant features could serve as potential predictors.

Outlier analysis indicates that `material_count` (138), `vertex_count` (117), `face_count` (122), `texture_count` (92), and `animation_count` (55)
contain substantial outliers, with `vertex_count` and `face_count` displaying extreme values far from the mean and median.

In other, `texture_count` and `material_count` have skewed distributions, with mean values higher than the median, while 
`animation_count` and `is_age_restricted` are almost constant.  

In terms of missing data, `pbr_type` is missing in approximately 77% of records and `texture_count` is missing in about 43% of records. All other columns are complete.  

Instead, in terms of extreme values, 90 models (~9%) have `face_count` greater than 200,000,
likely representing photogrammetry-derived models that are not representative of the expected target.

In addition, the ANOVA results show that `texture_count` does not vary significantly with `pbr_type` (p ≈ 0.092),
meaning it provides limited information about the `pbr_type` and cannot reliably substitute it.
In contrast, `material_count` is significantly associated with `pbr_type` (p ≈ 0.013),
suggesting it could serve as a potential proxy when `pbr_type` is missing, as it carries relevant information about the same property.

At the end, the most informative features for predicting `associated_tag` are `pbr_type` (if imputed),
`texture_count`, `vertex_count`, `material_count`, `face_count`, `user_tags`, and `user_categories`.
Careful handling of extreme values, missing data, and skewed distributions is necessary for robust modeling.