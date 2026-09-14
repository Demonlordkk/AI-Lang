import re
from dataclasses import dataclass
@dataclass(frozen=True)
class Token:
    kind:str; value:object; line:int; column:int
class LexError(Exception): pass
KEYWORDS={"let":"LET","var":"VAR","emit":"EMIT","fn":"FN","give":"GIVE","when":"WHEN","else":"ELSE","repeat":"REPEAT","in":"IN","record":"RECORD","use":"USE","to":"TO","true":"BOOL","false":"BOOL","done":"DONE","and":"AND","or":"OR","not":"NOT","Int":"TYPE","Real":"TYPE","Bool":"TYPE","Text":"TYPE","Byte":"TYPE"}
TWO={":=":"DEFINE","<-":"ASSIGN","->":"ARROW","==":"EQEQ","!=":"NE","<=":"LE",">=":"GE"}
ONE={"+":"PLUS","-":"MINUS","*":"STAR","/":"SLASH","%":"PERCENT","=":"EQUAL","<":"LT",">":"GT","(":"LPAREN",")":"RPAREN","[":"LBRACKET","]":"RBRACKET",",":"COMMA",":":"COLON",".":"DOT","?":"QUESTION"}
def lex(s):
    out=[];i=0;ln=1;col=1
    while i<len(s):
        c=s[i]
        if c in " \t\r": i+=1;col+=1;continue
        if c=="\n": i+=1;ln+=1;col=1;continue
        if c=="#":
            while i<len(s) and s[i]!="\n": i+=1;col+=1
            continue
        sl,sc=ln,col
        two=s[i:i+2]
        if two in TWO:
            out.append(Token(TWO[two],two,ln,col));i+=2;col+=2;continue
        if c in ONE:
            out.append(Token(ONE[c],c,ln,col));i+=1;col+=1;continue
        if c=='"':
            i+=1;col+=1;chars=[]
            while i<len(s) and s[i]!='"':
                if s[i]=='\\':
                    if i+1>=len(s): raise LexError(f"unterminated string at {sl}:{sc}")
                    esc=s[i+1];chars.append({'n':'\n','t':'\t','r':'\r','"':'"','\\':'\\'}.get(esc,esc));i+=2;col+=2
                else:
                    if s[i]=='\n': raise LexError(f"newline in string at {ln}:{col}")
                    chars.append(s[i]);i+=1;col+=1
            if i>=len(s): raise LexError(f"unterminated string at {sl}:{sc}")
            i+=1;col+=1;out.append(Token("TEXT","".join(chars),sl,sc));continue
        m=re.match(r"\d+(?:\.\d+)?",s[i:])
        if m:
            x=m.group();v=float(x) if '.' in x else int(x);out.append(Token("REAL" if isinstance(v,float) else "INT",v,ln,col));i+=len(x);col+=len(x);continue
        m=re.match(r"[A-Za-z_][A-Za-z0-9_]*",s[i:])
        if m:
            x=m.group();k=KEYWORDS.get(x,"IDENT");v=(x=="true") if k=="BOOL" else x
            out.append(Token(k,v,ln,col));i+=len(x);col+=len(x);continue
        raise LexError(f"unexpected character {c!r} at {ln}:{col}")
    out.append(Token("EOF",None,ln,col));return out
