# Data format

## Detection source

The default preparation script expects:

```text
detection_source/
├── GI/
│   ├── images/
│   └── labels/
├── PI/
│   ├── images/
│   └── labels/
└── CI/
    ├── images/
    └── labels/
```

Labels use normalized YOLO text rows:

```text
class_id center_x center_y width height
```

The public configuration maps the enclosing directory to type identifiers
GI=0, PI=1, and CI=2. The preparation script writes remapped labels to the
workspace and creates symbolic links to images. It never edits source files.

## Unlabelled SSL source

`ssl_source` may contain images recursively. Its label directories, if any,
are ignored. Supported extensions are JPEG, PNG, BMP, TIFF, and WebP.

The generated `ssl_list.txt` contains paths relative to the configured data
root. The training script also accepts a separately prepared text list.

## Fine-grained crop source

The default crop indexer expects existing train and validation directories:

```text
crop_source/
├── trian/                       # set probe.train_directory_name in config
│   ├── GI_N/
│   ├── GI_P/
│   ├── GI_S/
│   ├── PI_N/
│   ├── PI_P/
│   ├── PI_De/
│   ├── PI_Da/
│   ├── CI_N/
│   ├── CI_P/
│   ├── CI_Da/
│   └── CI_S/
└── val/
    └── same class directories
```

Set `probe.train_directory_name` to `train` for collections that use the
standard spelling. Files with a minimum side below 32 pixels are omitted. A
digest match between validation and training causes the training copy to be
omitted from the generated index. The original crop-level split is retained.

The generated CSV columns are:

```text
path,type,subclass,split
```

Only training rows are balanced. Within each type, all fine classes are
deterministically downsampled to the smallest class using random state 0.

Detector image lists use paths relative to a symbolic link located beside the
list file. This follows the Ultralytics list-file parser while keeping the
workspace portable across machines.

## Fine-grained full-image labels

End-to-end pipeline evaluation accepts YOLO labels whose class mapping uses:

```text
GI_N, GI_P, GI_S,
PI_N, PI_P, PI_De, PI_Da,
CI_N, CI_P, CI_Da, CI_S
```

Supply the mapping in a plain text file with one class name per class identifier.
