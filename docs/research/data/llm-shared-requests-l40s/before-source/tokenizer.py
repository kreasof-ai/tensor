"""LFM2's GGUF byte-level BPE tokenizer and bounded single-user chat formatting."""
from __future__ import annotations
import functools
import regex
from .gguf import GGUFError

PATTERN=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


class Tokenizer:
    def __init__(self, metadata):
        if metadata.get('tokenizer.ggml.model')!='gpt2' or metadata.get('tokenizer.ggml.pre')!='lfm2':
            raise GGUFError('requires the LFM2 byte-level BPE tokenizer')
        self.tokens=metadata['tokenizer.ggml.tokens'];self.ids={token:i for i,token in enumerate(self.tokens)}
        if len(self.ids)!=len(self.tokens):raise GGUFError('duplicate vocabulary token')
        self.ranks={tuple(pair.split(' ')):i for i,pair in enumerate(metadata['tokenizer.ggml.merges'])}
        self.special={token:i for i,token in enumerate(self.tokens) if metadata['tokenizer.ggml.token_type'][i] in (3,4)}
        self.bos=metadata['tokenizer.ggml.bos_token_id'];self.eos=metadata['tokenizer.ggml.eos_token_id']
        # Recognize the two released LFM2 single-user generation suffixes without
        # interpreting arbitrary Jinja. Thinking in historical assistant messages
        # does not imply that a new generation should start with <think>.
        template=metadata.get('tokenizer.chat_template')
        self.generation_prefix='<|im_start|>assistant\n<think>' if template is None else None
        if isinstance(template,str):
            match=regex.search(
                r'''\{%-?\s*if add_generation_prompt\s*-?%\}\s*\{\{-?\s*(["'])(<\|im_start\|>assistant\\n(?:<think>)?)\1\s*-?\}\}\s*\{%-?\s*endif\s*-?%\}\s*\Z''',
                template)
            if match:self.generation_prefix=match[2].replace('\\n','\n')
        values=list(range(33,127))+list(range(161,173))+list(range(174,256));characters=values.copy();extra=0
        for b in range(256):
            if b not in values:values.append(b);characters.append(256+extra);extra+=1
        self.encode_byte=dict(zip(values,map(chr,characters)));self.decode_byte={c:b for b,c in self.encode_byte.items()}
        self.pattern=regex.compile(PATTERN)
        self.special_pattern=regex.compile('|'.join(regex.escape(t) for t in sorted(self.special,key=lambda s:-len(s))))

    @functools.lru_cache(maxsize=8192)
    def _piece(self,text):
        encoded=''.join(self.encode_byte[b] for b in text.encode('utf-8'))
        # llama.cpp's LFM2 pre-tokenizer enables ignore_merges: whole vocabulary
        # pieces are emitted before attempting BPE, irrespective of merge rank.
        if encoded in self.ids:return (self.ids[encoded],)
        pieces=list(encoded)
        while len(pieces)>1:
            best=min(range(len(pieces)-1),key=lambda i:self.ranks.get((pieces[i],pieces[i+1]),float('inf')))
            if (pieces[best],pieces[best+1]) not in self.ranks:break
            pieces[best:best+2]=[pieces[best]+pieces[best+1]]
        try:return tuple(self.ids[p] for p in pieces)
        except KeyError as error:raise GGUFError('missing byte-level vocabulary token') from error

    def encode(self,text,*,add_bos=True,parse_special=False):
        if not isinstance(text,str):raise TypeError('text must be a string')
        result=[self.bos] if add_bos else []
        offset=0
        matches=self.special_pattern.finditer(text) if parse_special else ()
        for match in matches:
            for piece in self.pattern.findall(text[offset:match.start()]):result.extend(self._piece(piece))
            result.append(self.special[match.group()]);offset=match.end()
        for piece in self.pattern.findall(text[offset:]):result.extend(self._piece(piece))
        return result

    def decode(self,ids,*,skip_special=False):
        output=bytearray()
        for i in ids:
            if not 0<=int(i)<len(self.tokens):raise ValueError('invalid token ID')
            text=self.tokens[int(i)]
            if text in self.special:
                if not skip_special:output.extend(text.encode('utf-8'))
            else:output.extend(self.decode_byte[c] for c in text)
        return output.decode('utf-8',errors='replace')

    def chat(self,text):
        if not isinstance(text,str):raise TypeError('text must be a string')
        if self.generation_prefix is None:raise GGUFError('unsupported LFM2 chat generation prefix')
        # Initial profile: a single user turn, no tool declarations/history.
        # User text is encoded literally, so control strings cannot create roles.
        prefix=[self.bos,self.special['<|im_start|>']]+self.encode('user\n'+text,add_bos=False)
        suffix=self.encode('<|im_end|>\n'+self.generation_prefix,add_bos=False,parse_special=True)
        return prefix+suffix
