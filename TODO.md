# TODO
## Mats
- add wandb tracking
- add PrioritizedReplay Buffer
- improve logger to store the metrics and flush them in a structured way for activations, episodes and weights
- add early stopping
- add test environment (with one player)

## Training
- [x] check if action space is sound
- [x] grayscale images as input
- [ ] depth buffer, segmented buffer, ~~automap buffer~~
- [ ] implement and try out PPO
- [ ] Train a few agents
    - [ ] get one or two models as grading baselines
    - [ ] Basic reward (eg `FRAGS + 0.1 * HITCOUNT - 0.1 * HITS_TAKEN`)
    - [ ] reward tweaks
    - [x] Get baseline reward for grading

## Tutors
- [x] Check if it works in colab
- [x] Requirements and environment(to be tested)
- [x] Exercise notebook for students (basic training / model)
- [ ] Model saving/loading from ONNX (check [tournament.ipynb](/tournament.ipynb))
    - [ ] User config in yaml format
    - [ ] decide a configuration format so that each player can set up the a custom env in the tournament (eg custom n frames, different buffers)
    - [x] ONNX loading in graded / tournament

## GG
- [ ] Implement graded PvE script
- [x] Implement tournament PvP script
    - [x] Add custom colors and names for characters
    - [x] Include bots and adjust skill https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/cig_multiplayer_bots.py
    - [ ] Live leaderboard on wandb?
    - [x] Plotting function for tournament results
- [x] Create a custom `.wad` / `.cfg` for the tournament (start from cig or multi) - https://doom.fandom.com/wiki/Choosing_a_WAD_editor
    - [x] Also used to make bots slower
    - [x] Explore other maps / import them in `jku.wad`
