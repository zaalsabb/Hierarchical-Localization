import argparse
import contextlib
import io
import logging
import shutil
import sys
import multiprocessing
from pathlib import Path
import pycolmap

from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, geometric_verification)


def create_empty_db(database_path):
    if database_path.exists():
        logging.warning('The database already exists, deleting it.')
        database_path.unlink()
    logging.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(image_dir, database_path, camera_mode):
    logging.info('Importing images into the database...')
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    if isinstance(camera_mode, str):
        camera_mode = getattr(pycolmap.CameraMode, camera_mode)
    with pycolmap.ostream():
        pycolmap.import_images(database_path, image_dir, camera_mode)


def get_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(sfm_dir, database_path, image_dir):
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)
    logging.info('Running 3D reconstruction...')
    try:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            # TODO: stdout redirection hangs, probably due to a race condition.
            with pycolmap.ostream(stdout=False):
                reconstructions = pycolmap.incremental_mapping(
                    database_path, image_dir, models_path,
                    num_threads=min(multiprocessing.cpu_count(), 16))
    except Exception as e:
        logging.error('Reconstruction failed with output:\n%s', output)
        raise e
    sys.stdout.flush()

    if len(reconstructions) == 0:
        logging.error('Could not reconstruct any model!')
        return None
    logging.info(f'Reconstructed {len(reconstructions)} model(s).')

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logging.info(f'Largest model is #{largest_index} '
                 f'with {largest_num_images} images.')

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        shutil.move(
            str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def main(sfm_dir, image_dir, pairs, features, matches,
         camera_mode=pycolmap.CameraMode.AUTO,
         skip_geometric_verification=False, min_match_score=None):

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    import_images(image_dir, database, camera_mode)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(database, pairs)
    reconstruction = run_reconstruction(sfm_dir, database, image_dir)
    if reconstruction is not None:
        logging.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                     + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--camera_mode', type=str, default="AUTO",
                        choices=list(pycolmap.CameraMode.__members__.keys()))
    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    args = parser.parse_args()

    main(**args.__dict__)
