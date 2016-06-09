# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:46:43 2016

@author: mbochk
"""

"""
Programm lists all imports from correct python programm
"""

import re


def search_results(pattern, text):
    for m in re.finditer(pattern, text):
        print "’{0}’: {1}-{2}".format(m.group(), m.start(), m.end())


def import_patterns():
    module_name_pattern = "\w+(\.\w+)*"
    next_module_name = "\s*,\s*" + module_name_pattern
    several_modules_patern = module_name_pattern + "(" + next_module_name \
        + ")*"
    semilicon_ending_pattern = "\s*;?"

    simple_pattern = "import" + "\s+" + several_modules_patern \
                     + semilicon_ending_pattern
    from_pattern = "from" + "\s+" + module_name_pattern \
                   + "\s+" + "import" + "\s+" + several_modules_patern

    patterns = [from_pattern, simple_pattern]
    unstripped_pattern = ["\s*" + r"\b" + pattern + semilicon_ending_pattern
                          for pattern in patterns]

    return [re.compile(pattern) for pattern in unstripped_pattern]


def extract_import(group):
    words = re.split("[\s,;]", group.strip())
    max_index = len(words)
    if "from" in words:
        max_index = words.index("import")
    return [word for word in words[1:max_index] if len(word) > 0]


def add_imports(patterns, line, import_set):
    while len(line) > 0:
        last_pos = len(line)
        for pattern in patterns:
            match = pattern.match(line)
            if match is not None:
                for word in extract_import(match.group()):
                    import_set.add(word)
                last_pos = match.end()
                break
        line = line[last_pos:]
        # print line


def main():
    filename = "input.txt"
    patterns = import_patterns()
    import_list = set()
    with open(filename, 'r') as f:
        for line in f:
            # print line.strip()
            add_imports(patterns, line.strip(), import_list)
    import_list = sorted([module for module in import_list])
    print ", ".join(import_list)

if __name__ == "__main__":
    main()
