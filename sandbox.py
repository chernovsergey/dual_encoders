# -*- coding: utf-8 -*-

import re
import string

def remove_punctuation(text):
    return re.sub(ur"\p{P}+", "", text)

def remove_punctuation(to_translate, translate_to=u''):
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
    return to_translate.translate(translate_table)

s = u'аьаьавыдалоываБЮЁ№!"№!"№"!ЮБ,бю.,'

print remove_punctuation(s)