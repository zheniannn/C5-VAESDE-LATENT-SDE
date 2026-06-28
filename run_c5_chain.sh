set -e
VPY=/home/ian/.venvs/venv/bin/python
CFG=configs/latent_sde_default.yaml
echo "### C5 TRAIN ###";   $VPY scripts/run_train_latent_sde.py --config $CFG
echo "### C5 SCORE ###";   $VPY scripts/run_score_latent_sde.py --config $CFG
echo "### C5 STRESS p99 ###"; $VPY scripts/run_stress_test_latent_sde.py --config $CFG --score-name total_nll --quantile 0.99
echo "### C5 STRESS p95 ###"; $VPY scripts/run_stress_test_latent_sde.py --config $CFG --score-name total_nll --quantile 0.95
echo "### C5 ROLLOUT ###"; $VPY scripts/run_rollout_latent_sde.py --config $CFG
echo "### C5 CHAIN COMPLETE ###"
