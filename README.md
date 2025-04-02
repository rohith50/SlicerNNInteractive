# `nnInteractiveSlicer`

## Installation

`nnInteractiveSlicer` needs to be set up on the server-side and the client-side. The server-side needs relatively heavy compute, as described here:

> You need a Linux or Windows computer with a Nvidia GPU. 10GB of VRAM is recommended. Small objects should work with <6GB. nnInteractive supports Python 3.10+
> 
> -- [The nnInteractive README](https://github.com/MIC-DKFZ/nnInteractive?tab=readme-ov-file#prerequisites)

The client machine _can_ be the same as the server machine.

### Server-side

You can install the server-side of `nnInteractiveSlicer` in two different ways:

#### Option 1: Using Docker

```
docker pull coendevente/nninteractive-slicer-server:latest
docker run --gpus all --rm -it -p 1527:1527 coendevente/nninteractive-slicer-server:latest
```

This will make the server available under port `1527` on your machine. If you would like to use a different port, say `1627`, replace `-p 1527:1527` with `-p 1627:1527`.

#### Option 2: Using `pip`

```
pip install nninteractive-slicer-server
nninteractive-slicer-server --host 0.0.0.0 --port 1527
```

If you would like to use a different port, say `1627`, replace `--port 1527` with `--port 1627`.

### Client-side: Installation in 3D Slicer

For now, `nnInteractiveSlicer` is not yet available in the Extensions Manager of 3D Slicer. So, currently, the following steps are still needed to install the `nnInteractiveSlicer` extension on the client-side in 3D Slicer:

1. `git clone git@github.com:coendevente/nninteractive-slicer.git` (or download the current project as a `.zip` file from GitHub).
2. Open 3D Slicer and click the Module dropdown menu in the top left of the 3D Slicer window:
	![Slicer dropdown menu](img/dropdown.png)
3. Go to `Developer Tools` > `Extension Wizard`.
4. Click `Select Extension`.
5. Locate the `nninteractive-slicer` folder you obtained in Step 1, and select the `slicer_plugin` folder and click "Open".
6. Go to the Module dropdown menu again and go to `Segmentation` > `nnInteractiveSlicer`. This should result in the following view:
  ![First view of the Slicer extension](img/plugin_first_sight.png)
7. Configure the right server settings by going to the `Configuration` tab. Then type in the URL of the server you set up in the [server-side](#server-side) installation procedure. This should look something like `http://remote_host_name:1527` or, if you run the server locally, `http://localhost:1527`.

## Usage

## Citation

## License
