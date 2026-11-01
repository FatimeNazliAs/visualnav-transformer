# Robot search videos

A small wheeled robot is placed in a 3D-scanned home and must find one kind of object: a bed, chair, sofa, toilet,
TV or plant. Each video is one drive of up to 1,000 steps (4.2 minutes, shown in real time). The homes come from two
collections: **HM3D** (our main test homes) and **MP3D** (a second collection, never seen in training).

## Folders

| Folder | What drives the robot |
|---|---|
| `0_start_here/` | 6 videos picked because they are easy to follow. **Not representative**: they do not show how often each robot succeeds. |
| `1_MapMaD_with_map/` | Our model, MapMaD: sees the camera picture plus a top-down map it builds while driving. `v2.1` and `v2.1b` are two training versions (v2.1b practised equally with and without the map). |
| `2_MapMaD_map_hidden/` | The same model with the map blanked out, to show what the map adds. |
| `3_NoMaD_original/` | The published model we started from: camera only, no map. |
| `4_Frontier_planner_F/` | A hand-written program on the same map: drive to the nearest unexplored edge, go to the object once the map marks it. The strongest robot here. |
| `5_VLFM_style_VF/` | Like F, but an image–text model (BLIP-2) scores each direction for "how likely is the object there". A copy of the published method VLFM adapted to our robot. |

## The screen

- **Left, camera:** orange boxes = the object detector (YOLOE) thinks this is the target (number = its confidence);
  green boxes = where the target really is (from the simulator).
- **Right, map:** the whole floor seen from above; blue line = the robot's path, white arrow = the robot, green squares =
  the real targets with their green "good viewing spots".
- **Bottom right, timeline:** the whole drive; coloured marks = events (first sight, first detector box, reached,
  stop). The white cursor shows where the video is now.

## File names

`<HM3D|MP3D>_<target>_<did it reach>_<what the stop rule did>[_v2.1|_v2.1b].mp4`

- `reaches-bed`: came within 0.2 m of a good viewing spot of a real bed: **this counts as success**.
  `sees-bed-never-reaches`: the bed was in view but the robot never got there. `never-sees-bed`: never in view.
- The **stop rule** says "stop here" once the map is sure enough: `stops-at-it` (right object), `stops-at-a-wrong-object`,
  `stops-near-it-not-close-enough`, `stops-in-a-wrong-place`, `never-stops`. The robot keeps driving after that
  moment, so both outcomes are shown in one video.
- `_2` = a second drive with the same description. `index.csv` lists the original file and episode of every video.

## The 4 things to check

1. **Where is the target?** Green boxes in the camera, green squares on the map.
2. **Did the detector see it?** An orange box on top of a green box ("1st YOLOE box on a true target"). "1st YOLOE box
   (any object)" is the detector's first box on anything.
3. **When was the drive scored?** A 2-second freeze with a green frame: "at a target view spot".
4. **Was the stop on the right object?** A 2-second freeze titled "rule stop: right object / wrong object / …".

## What "dimmed" means

After the robot reaches the target (the success moment), the video freezes for 2 seconds and then shows the rest of
the drive darker, with the note "after the scored moment … dimmed". Nothing in the dimmed part changes the success
score; it is shown only so you can see what the robot did next.

Boxes carry the target's name (for example "sofa"); the detector was asked with its own word, shown in the legend
("YOLOE prompt: 'couch'").
