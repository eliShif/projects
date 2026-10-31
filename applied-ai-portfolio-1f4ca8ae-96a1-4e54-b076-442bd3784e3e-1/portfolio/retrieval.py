"""Local retrieval over an explicit corpus; no arbitrary URLs or file access."""
from __future__ import annotations
from dataclasses import dataclass
import ast, operator
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Document:
    id: str
    text: str


class Retriever:
    def __init__(self,documents):
        self.documents=list(documents)
        if not self.documents or len({d.id for d in self.documents})!=len(self.documents):
            raise ValueError('Need a nonempty corpus with unique IDs')
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
        self.matrix=self.vectorizer.fit_transform([d.text for d in self.documents])

    def search(self,query,k=3):
        if not isinstance(query,str) or not query.strip() or len(query)>4000:raise ValueError('Invalid query')
        if not isinstance(k,int) or not 1<=k<=10:raise ValueError('Invalid k')
        scores=(self.matrix@self.vectorizer.transform([query]).T).toarray().ravel()
        order=np.argsort(-scores,kind='stable')[:k]
        return [{'id':self.documents[i].id,'text':self.documents[i].text,'score':float(scores[i])} for i in order if scores[i]>0]


def calculate(expression):
    """Small numeric DSL. No eval, names, attributes, calls or unbounded powers."""
    if not isinstance(expression,str) or len(expression)>100:raise ValueError('Expression too long')
    tree=ast.parse(expression,mode='eval')
    if sum(1 for _ in ast.walk(tree))>30:raise ValueError('Expression too complex')
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):value=node.value
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=visit(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
        elif isinstance(node,ast.BinOp) and type(node.op) in ops:
            value=ops[type(node.op)](visit(node.left),visit(node.right))
        else:raise ValueError('Only numeric arithmetic is allowed')
        if not np.isfinite(value) or abs(value)>1e12:raise ValueError('Numeric result out of bounds')
        return value
    try:return float(visit(tree.body))
    except (ZeroDivisionError,OverflowError) as e:raise ValueError('Invalid arithmetic') from e
