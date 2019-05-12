{ time python -u experiment.py --sc-root ~/datasets/speech_commands --experiments-root ~/experiments --run compare_ae_vae_source_label ; } |& tee ~/experiments/compare_ae_vae_source_label/experiment.log
{ time python -u experiment.py --sc-root ~/datasets/speech_commands --experiments-root ~/experiments --run vary_num_classes ; } |& tee ~/experiments/vary_num_classes/experiment.log
