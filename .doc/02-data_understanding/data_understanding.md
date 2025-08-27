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
