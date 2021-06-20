"""
Adaptor for reading ImJoy Annotation datasets.

This can either read a CSV file with labeled frames for a single video,
or a YAML file which potentially contains multiple videos.

The adaptor was created by manually inspecting DeepLabCut files and there's no
guarantee that it will perfectly import all data (especially metadata).

If the adaptor can find full video files for the annotated frames, then the
full videos will be used in the resulting SLEAP dataset. Otherwise, we'll
create a video object which wraps the individual frame images.
"""

import os
import re
import json

import numpy as np
import pandas as pd

from typing import List, Optional

from sleap import Labels, Video, Skeleton
from sleap.instance import Instance, LabeledFrame, Point
from sleap.util import find_files_by_suffix
from shapely.geometry import LineString

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class LabelsImJoyAdaptor(Adaptor):
    """
    Reads ImJoy annotation folder with labeled frames for single video.
    """

    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "json"

    @property
    def all_exts(self):
        return ["json"]

    @property
    def name(self):
        return "ImJoy Annotation Dataset JSON"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename) or not file.filename.endswith('manifest.json'):
            return False
        # TODO: add checks for valid deeplabcut csv
        return True

    def can_write_filename(self, filename: str):
        return False

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return False

    @classmethod
    def read(
        cls,
        file: FileHandle,
        full_video: Optional[Video] = None,
        *args,
        **kwargs,
    ) -> Labels:
        return Labels(
            labeled_frames=cls.read_frames(
                file=file, full_video=full_video, *args, **kwargs
            )
        )

    @classmethod
    def make_video_for_image_list(cls, image_dir, filenames) -> Video:
        """Creates a Video object from frame images."""

        # the image filenames in the csv may not match where the user has them
        # so we'll change the directory to match where the user has the csv
        def fix_img_path(img_dir, img_filename):
            img_filename = img_filename.replace("\\", "/")
            # img_filename = os.path.basename(img_filename)
            img_filename = os.path.join(img_dir, img_filename)
            return img_filename

        filenames = list(map(lambda f: fix_img_path(image_dir, f), filenames))

        return Video.from_image_filenames(filenames)

    @classmethod
    def read_frames(
        cls,
        file: FileHandle,
        skeleton: Optional[Skeleton] = None,
        # full_video: Optional[Video] = None,
        *args,
        **kwargs,
    ) -> List[LabeledFrame]:
        filename = file.filename
        root = os.path.dirname(filename)
        # Read JSON
        with open(filename, "r") as f:
            data = json.loads(f.read())
        
        node_names = [str(n) for n in range(10)]

        if skeleton is None:
            skeleton = Skeleton()
            skeleton.add_nodes(node_names)
            for i in range(len(node_names)-1):
                skeleton.add_edge(node_names[i], node_names[i+1])

        # Get list of all images filenames.
        samples = data["samples"]

        img_files = [os.path.join(s["sample_id"], 'image.png') for s in samples]
        # Create the Video object
        img_dir = os.path.dirname(filename)
        video = cls.make_video_for_image_list(img_dir, img_files)

        lfs = []
        for i in range(len(samples)):
            with open(os.path.join(img_dir, samples[i]["sample_id"], 'target_files_v1', 'annotation.json'), 'r') as f:
                features = json.loads(f.read())["features"]
            coords = [f["geometry"]["coordinates"] for f in features if f["geometry"]["type"]=='LineString']
            instances = []
            for coord in coords:
                line = LineString(coord)
                ll = line.length
                # Get points for each node.
                instance_points = dict()
                for j, node in enumerate(node_names):
                    p = line.interpolate(ll/len(node_names)*j)
                    instance_points[node] = Point(p.x, p.y)

                # Create instance with points assuming there's a single instance per
                # frame.
                instances.append(Instance(skeleton=skeleton, points=instance_points))

            # Create LabeledFrame and add it to list.
            lfs.append(
                LabeledFrame(video=video, frame_idx=i, instances=instances)
            )

        return lfs
