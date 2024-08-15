# Special-purpose tree

This is a branch of NeVA for experimenting using Thunder:

	https://github.com/Lightning-AI/lightning-thunder

Unless you work with the thunder team, you should be using upstream NeMo:

	https://github.com/NVIDIA/NeMo

# Scripts overview

* `neva.sh` -- for testing the thunder-only path for NeVA
* `dynamo-neva.sh` -- for testing the Dynamo+Thunder path for NeVA

To use eager mode, you can just remove the setting for the
`NEMO_THUNDER_NEVA` environment variable in the scripts.
