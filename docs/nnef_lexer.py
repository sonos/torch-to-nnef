from pygments.lexer import RegexLexer, bygroups
from pygments.token import *


class NNEFLexer(RegexLexer):
    name = "NNEF"
    aliases = ["nnef"]
    filenames = ["*.nnef", "*.quant"]

    # (r';.*', Comment),
    # (r'\[.*?\]$', Keyword),
    # (r'(.*?)(\s*)(=)(\s*)(.*)',
    #
    tokens = {
        "root": [
            (r"#.*", Comment),
            (";", Punctuation),
            (r"(true|false)", bool),
            (
                r"(?<![a-zA-Z:])[-+]?\d*\.?\d+",
                Number,
            ),
            (r"\".*\"", String),
            (r"\'.*\'", String),
            (
                r"([\w_]+)( +)(=)( +)([\w_]+)(\()",
                bygroups(Name, Whitespace, Text, Whitespace, Operator, Text),
            ),
            (
                r"([\w_]+)( +)(=)( +)([\w_]+)(\<)",
                bygroups(Name, Whitespace, Text, Whitespace, Operator, Text),
            ),
            (r"^(fragment|graph|version|extension) ", Keyword),
        ]
    }
