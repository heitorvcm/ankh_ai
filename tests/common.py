"""Utilidades dos testes: registra um 'test_5p' (basic_5p com os 2 monumentos de coordenada invalida
movidos para casas validas) apenas para os testes; o arquivo ankh_map.py do usuario nao e alterado."""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ankh_map


def register_test_5p():
    sc = copy.deepcopy(ankh_map.SCENARIOS["basic_5p"])
    sc["monuments"] = [(t, (9, 4) if rc == (9, 3) else (9, 8) if rc == (9, 9) else rc, o)
                       for t, rc, o in sc["monuments"]]
    ankh_map.SCENARIOS["test_5p"] = sc
    return "test_5p"
