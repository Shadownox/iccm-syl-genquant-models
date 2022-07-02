from ccobra.benchmark.runner import entry_point
import sys

sys.argv.append('benchmark/prediction_genquant_test.json')
sys.exit(entry_point())