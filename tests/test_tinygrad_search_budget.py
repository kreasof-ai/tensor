"""CPU tests for retaining valid search results at a time/correctness boundary."""
import json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
try:
    from benchmarks.lfm2.tinygrad_search_budget import BudgetSearch,SearchDeadline,search
except ModuleNotFoundError:
    BudgetSearch=None


class FakeOutput:
    def __init__(self):self.allocator=SimpleNamespace(_copyin=self.write);self._buf=None
    def write(self,unused,data):self.value=np.frombuffer(data,np.float32).copy()
    def numpy(self):return self.value


@unittest.skipUnless(BudgetSearch,'requires the optional pinned tinygrad experiment environment')
class SearchBudgetTests(unittest.TestCase):
    def make_budget(self,path):
        expected=np.array([[1,2],[3,4]],np.float64)
        budget=BudgetSearch(60,path,np.zeros((2,2),np.float32),np.zeros((2,2),np.float16),expected,np.full((2,2),1e-6))
        budget.started=time.perf_counter()
        return budget,FakeOutput()

    def test_partial_output_cannot_reuse_the_previous_winner(self):
        with tempfile.TemporaryDirectory() as path:
            budget,output=self.make_budget(path)
            output.value=np.array([1,2,3,4],np.float32)
            def incomplete(*args,**kwargs):output.value[0]=1;return [1e-9]
            budget.original_time=incomplete
            times=budget.timed(None,{},[output])
            self.assertTrue(np.isinf(times).all())
            self.assertIsNone(budget.best)
            self.assertEqual(json.loads((Path(path)/'search-progress.json').read_text())['rejected'],1)

    def test_correct_candidate_is_checkpointed_before_deadline(self):
        with tempfile.TemporaryDirectory() as path,patch('benchmarks.lfm2.tinygrad_search_budget.diskcache_put') as cache:
            budget,output=self.make_budget(path)
            program=SimpleNamespace(src=[None,None,SimpleNamespace(arg='shader'),SimpleNamespace(arg=b'binary')])
            candidate=SimpleNamespace(applied_opts=['legal']);candidate.copy=lambda:candidate
            budget.programs[b'binary']=candidate;budget.key={'key':'candidate'}
            def complete(*args,**kwargs):output.value[:]=[1,2,3,4];return [3e-6,2e-6]
            budget.original_time=complete
            budget.timed(program,{},[output])
            self.assertEqual(budget.best[1],2e-6);cache.assert_called_once()
            self.assertTrue((Path(path)/'best-source.txt').exists())
            budget.started-=61
            with self.assertRaises(SearchDeadline):budget.timed(program,{},[output])
            self.assertEqual(budget.best[1],2e-6)

    def test_hooks_restore_after_interruption(self):
        originals=(search._try_compile,search._time_program,search.beam_search)
        with tempfile.TemporaryDirectory() as path:
            budget,_=self.make_budget(path)
            with self.assertRaises(SearchDeadline):
                with budget:raise SearchDeadline()
        self.assertEqual((search._try_compile,search._time_program,search.beam_search),originals)


if __name__=='__main__':unittest.main()
