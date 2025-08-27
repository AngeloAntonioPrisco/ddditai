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
- Multithreading to speed up collection  

**Metadata to Extract:**  

<div style="display: table; margin: auto;">

| Feature           | Description                                                      |
|-------------------|------------------------------------------------------------------|
| uid               | Unique identifier of the model.                                  |
| is age restricted | Boolean indicating if the model has age restriction.             |
| pbr type          | Declared PBR type, can be an empty string if not defined.        |
| texture count     | Number of declared textures.                                     |
| vertices          | Number of declared vertices of the model.                        |
| materials         | Number of declared materials of the model.                       |
| animations        | Number of declared animations associated with the model.         |
| user tags         | List of user-defined tags attached to the model.                 |
| user categories   | Categories assigned to the model.                                |
| face count        | Number of declared faces. Considered a substitute for triangles. |
| material count    | Number of declared materials in the model.                       |

</div>

**Storage Plan:**  
- Data to be logged and versioned in MLflow  
- Exportable to CSV for offline analysis