"""Bounded output-history proposals; every proposal still requires verification."""
from collections import OrderedDict


class OutputLookup:
    def __init__(self,*,min_context=3,max_context=8,confirmations=2,capacity=32768):
        if not (1<=min_context<=max_context and confirmations>=1 and capacity>=1):
            raise ValueError('invalid output lookup configuration')
        self.min_context,self.max_context=min_context,max_context
        self.confirmations,self.capacity=confirmations,capacity
        self.history=[];self.entries=OrderedDict()

    def append(self,token):
        token=int(token)
        for size in range(self.min_context,min(self.max_context,len(self.history))+1):
            key=tuple(self.history[-size:]);old=self.entries.get(key)
            count=old[1]+1 if old is not None and old[0]==token else 1
            self.entries[key]=(token,count);self.entries.move_to_end(key)
            if len(self.entries)>self.capacity:self.entries.popitem(last=False)
        self.history.append(token)
        self.history=self.history[-self.max_context:]

    def propose(self,count):
        if count<0:raise ValueError('invalid proposal count')
        history=self.history.copy();result=[]
        for _ in range(count):
            token=None
            for size in range(min(self.max_context,len(history)),self.min_context-1,-1):
                value=self.entries.get(tuple(history[-size:]))
                if value is not None and value[1]>=self.confirmations:
                    token=value[0];break
            if token is None:return None
            result.append(token);history.append(token)
        return result
