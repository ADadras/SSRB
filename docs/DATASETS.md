# Datasets and preparation

For reproduction, keep acquired data, derived images/masks, manifests containing identifiers, and weights outside Git. Store converter outputs outside the repository. They do not grant redistribution rights. Use the same authorized dataset release and split files as the paper. The single LIDC illustration in the README has separate [attribution and terms](../figures/README.md).

## BSDS500

Obtain the [Berkeley BSDS500 release](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html). Point `prepare-bsds --data-root` to the directory containing `images/{train,val,test}` and `groundTruth/{train,val,test}`. The converter considers regions from every human partition, keeps area fractions from 0.015 through 0.85, sorts by decreasing area, merges near-duplicates at Dice >=0.95 and retains at most 16 masks. Images need at least two retained masks. The accepted conversion has 199/100/200 train/validation/test images.

This is a task about alternative groupings of image regions. It differs from multiple annotations of one medical object. Dataset assets are not included under the MIT code license.

## IMA++

Obtain the masks, metadata and official train/val/test CSVs from [IMA++](https://github.com/sfu-mial/IMAplusplus) and its [Zenodo record](https://zenodo.org/records/14201693). Obtain the matching images from the ISIC Archive under their individual terms. The repository's Apache license does not establish a blanket license for source images.

Input layout for `prepare-ima --data-root "$IMA" --output "$SOURCE"`:

```text
$IMA/raw/train.csv, val.csv, test.csv
$IMA/raw/seg_metadata_multiannotator_subset.csv
$IMA/segs/<seg_filename>
$IMA/images/<ISIC_id>.jpg
```

The converter uses annotators beginning with A, orders masks by `seg_filename`, and keeps tasks with at least two masks. Use its `all_annotations.json` for the fixed visible choices. No acquisition or medical-data redistribution is performed by this command.

## RIGA/RIGA+

Obtain the depositor's `RIGAPlus.zip` and review the [original terms](https://deepblue.lib.umich.edu/data/downloads/g732d957m). Redistribution is restricted; a mirror's license label is insufficient. Run `prepare-riga --zip-path "$RIGA_ZIP" --output "$SOURCE"`.

The converter reads the ten dataset train/test CSVs and six grader masks. Disc is `value >= 128`, cup is `value == 255`. Test CSVs define test membership; SHA1 of the source image path places 20% of training CSV images into validation. The two tasks share the source fundus split but use separate models. Freeze `riga_disc.json` and `riga_cup.json` in separate directories for training. For exact frozen features, the converter also writes `feature_images.json` with both tasks and image metadata only. Extract features from this shared manifest before filtering each model's cache. This preserves the original batch boundaries, including the final batch of 14 training entries.

## LIDC-IDRI

Use the [TCIA LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri/) and the QIICR DICOM-SEG annotations used by the accepted conversion. TCIA lists a CC BY 3.0 license and data citation requirements. Do not commit patient identifiers, DICOM headers, images, masks or derived manifests.

Input layout for `prepare-lidc --data-root "$LIDC" --output "$SOURCE"`:

```text
$LIDC/raw/seg_qiicr/<series>/*.dcm
$LIDC/raw/ct_slices/<SOPInstanceUID>.dcm
```

The original acquisition used the TCIA NBIA `getSeries` API for Collection=LIDC-IDRI, retaining Modality=SEG, Manufacturer=QIICR and descriptions beginning with “Segmentation of Nodule”. Each selected SEG series was retrieved with `getImage` and placed under its patient directory. The API base is `https://services.cancerimagingarchive.net/nbia-api/services/v1`. Obtain referenced CT slices through TCIA and name them by their SOPInstanceUID as shown above. The converter reports a missing slice instead of downloading it. The method groups annotations by patient/study/nodule, chooses the central slice by summed annotated area, and may use the closest annotated plane within 1.5 mm where a grader lacks that slice. At least two masks must remain. It clips CT intensities to [-1000,500] HU. The square crop comes from the union of masks, padded by 24 pixels, with minimum side 96. This supplies annotation-derived location information before visible masks are chosen. Crop construction is separate from training. Full-image nodule localization was not evaluated.

SHA1 of patient ID defines the 70%/15%/15% patient split. The evaluator alone derives LIDC-hard by retaining test references with visible/reference Dice <0.80. It is not a separate dataset or training selector.

## Freeze the inputs

For a converter's `all_annotations.json` (or one RIGA task JSON):

```bash
python -m ssrb.cli make-visible-manifests --source "$SOURCE/all_annotations.json" --data-root "$SOURCE" --output "$PREPARED" --seed 0
```

The generic source JSON contains `image_id`, `source_id`, `endpoint`, `split`, `image_path` and an ordered `mask_paths` list. Paths are relative to the source directory. Freeze once per visible seed, preserve the resulting hashes, and keep preprocessing files away from the trainer. See REPRODUCE_PAPER.md for an end-to-end preparation example.
