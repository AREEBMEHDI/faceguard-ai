# Known Faces Directory

This directory holds reference photos for each person the system should recognise.

## Structure

```
known_faces/
├── person_name_1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person_name_2/
│   └── photo1.jpg
└── ...
```

## Guidelines

| Rule | Detail |
|------|--------|
| **One folder per person** | Folder name becomes the display label in alerts |
| **3–10 photos recommended** | Variety: different angles, lighting, distances |
| **JPEG or PNG** | Any resolution; system resizes internally |
| **No subfolders** | Place images directly inside the person's folder |

## Privacy

Face images are intentionally excluded from version control (`.gitignore`).  
Never commit real biometric data to a public repository.

## Running `load_known_faces`

The recognizer calls `load_known_faces()` automatically at startup.  
To reload without restarting, call the function again in your code.
