from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
from math import log1p
from pathlib import Path
import re

import torch


BOOK_PATH = Path(__file__).with_name('wizard_of_oz.txt')
READER_MODEL = 'deepset/minilm-uncased-squad2'
READER_REVISION = '934656cdda79824eabf503ed56e15c01ddbdbe3f'
NO_ANSWER = "I couldn't find an answer supported by the book."
PASSAGE_WORDS = 165
PASSAGE_STEP = 100
PASSAGE_CONTEXT_WORDS = 24
SEARCH_LIMIT = 8
EXPANDED_SEARCH_LIMIT = 24
FOLLOWUP_RECHECK_MARGIN = 5.0
EXPANDED_MATCH_FRACTION = 0.75
MIN_ANSWER_MARGIN = 2.0
MIN_SINGLE_PASSAGE_MARGIN = 3.25
MIN_ROUTE_CONFIRMATION_MARGIN = 6.0
ANSWER_AGREEMENT_BONUS = 0.5
RETRIEVAL_SCORE_WEIGHT = 0.5
ANSWER_SCORE_TOLERANCE = 0.02
DIRECTION_SCORE_TOLERANCE = 0.25
MIN_RETRY_TERMS = 2
CHAPTER_HEADING_WEIGHT = 0.28

CHAPTER_HEADING = re.compile(r'CHAPTER (\d+)\.', re.IGNORECASE)
SENTENCE_END = re.compile(r"""[.!?]["'’”_\])]*$""")
ABBREVIATION = re.compile(r'(?:[A-Z]|Mr|Mrs|Ms|Dr|St)\.$')
SHORT_FOLLOW_UP = re.compile(
    r'^(?:why|how many|what kind|which one)\W*$|^why (?:that|this) name\W*$'
    r'|\b(?:there|that|those|too|else)\W*$', re.IGNORECASE,
)
PRONOUN = re.compile(r'\b(?:he|she|it|they|him|her|them|his|hers|its|their)\b', re.IGNORECASE)
FEMALE_NAMES = r'Jellia(?: Jamb)?|Dorothy|Eureka|Ozma|Aunt Em|Billina'
MALE_NAMES = (
    r'Braided Man|Cowardly Lion|Hungry Tiger|Tin Woodman|Woggle-Bug|Nick Chopper|'
    r'Uncle Henry|Uncle Bill|Gwig|Jim|Sawhorse|Scarecrow|Tik-tok|Wizard|Zeb(?:ediah)?|Captain-General'
)
BOOK_NAMES = f'{FEMALE_NAMES}|{MALE_NAMES}'
BOOK_NAME = re.compile(rf'\b(?:{BOOK_NAMES})\b', re.IGNORECASE)
FEMALE_NAME = re.compile(rf'(?:{FEMALE_NAMES})', re.IGNORECASE)
MALE_NAME = re.compile(rf'(?:{MALE_NAMES})', re.IGNORECASE)
NAME = rf"(?:(?i:the) )?(?:(?i:{BOOK_NAMES})\b|[A-Z][\w'-]+(?: [A-Z][\w'-]+){{0,3}})"
NAMED_SUBJECT = re.compile(rf'(?i:who|what) (?i:is|was) ({NAME})')
QUESTION_ACTOR = re.compile(
    rf"(?i:who|what|where|when|why|how(?: long| much| high| far| often)?) "
    rf"(?i:did|does|was|were|is|has|had|could|would) ({NAME})(?= |'s\b|$)",
)
COORDINATED_ACTOR = re.compile(rf'(?i: and) ({NAME})(?= |$)')
PLURAL_PERSON = re.compile(r'\b(?:they|them|their)\b', re.IGNORECASE)
SPEECH_TOPIC = re.compile(
    rf"(?i:what did) (?P<speaker>{NAME}) (?:(?i:say)|(?i:tell) (?P<listener>{NAME})) "
    r"(?i:about) (?P<topic>.+?)(?: (?i:during|when|while|after|before) .+)?[?.!]*$",
)
GROUP_REQUEST = re.compile(rf"(?i:what did) ({NAME}) (?i:ask (?:the others|everyone|them) to do when) (.+?)\W*$")
REQUEST_REASON = re.compile(r"why did (?:he|she|they) (?:want|suggest) (?:it|that)\W*", re.IGNORECASE)
SPEECH_VERB = r'said|declared|answered|replied|asked|murmured|cried|added|exclaimed|returned|growled|grumbled|explained'
SPEECH_ALIASES = {'ozma': 'the Princess', 'jim': 'the horse'}
TAGGED_SPEAKER = rf"(?:{NAME}|{'|'.join(map(re.escape, SPEECH_ALIASES.values()))})"
SPEECH_TAG = re.compile(
    rf'(?:(?:{SPEECH_VERB}) (?P<after>{TAGGED_SPEAKER})|(?P<before>{TAGGED_SPEAKER}) (?:{SPEECH_VERB}))(?=[,.; ]|$)',
)
DESCRIPTION = re.compile(r"(?:it is|it's) (.+?)\W*", re.IGNORECASE)
SPEECH_QUOTE = re.compile(r'"[^"]+"')
PROMISE = re.compile(rf'(?i:what did) ({NAME}) (?i:promise (?:him|her|them))\W*')
APPEARANCE = re.compile(r"how did ((?:the|this|that) [a-z][\w -]*?) (?:look|seem)\b", re.IGNORECASE)
DISTRESS = re.compile(r"\b(?:afraid|alarmed|frightened|scared|terrified|worried)\b", re.IGNORECASE)
TIME_CLAUSE = re.compile(r'\b(?:when|while|after|before)\b', re.IGNORECASE)
REPORTING_ACTOR = re.compile(
    rf'(?i:\b(?:did|does)) ({NAME}|(?i:the [a-z-]+(?: [a-z-]+){{0,3}})) (?i:say|claim|report)\b',
)
REPORTED_AGE = re.compile(
    r'how old did (he|she|it|they) (?:say|claim|report) \1( and (?:his|her|its|their) [a-z]+)? '
    r'(?:was|were|(?:had )?(?:just )?turned)\W*', re.IGNORECASE,
)
AGE_ANSWER = re.compile(r'(?:[a-z0-9]+(?:[- ][a-z0-9]+){0,4}) years? old', re.IGNORECASE)
COLOR_QUESTION = re.compile(r'what color (is|are|was|were) (.+?)\W*', re.IGNORECASE)
DAMAGE_REFERENCE = re.compile(r'what caused that damage(?: in the first place)?\W*', re.IGNORECASE)
DAMAGE_VERBS = {'cracked': 'crack', 'broke': 'break', 'shattered': 'shatter', 'collapsed': 'collapse'}
DAMAGE_CLAUSE = re.compile(
    r"\bwhen ((?:the|a|an) [\w'-]+(?: [\w'-]+){0,3}) "
    r'(crack|cracked|break|broke|shatter|shattered|collapse|collapsed|split)$', re.IGNORECASE,
)
REPAIR_FOLLOWUP = re.compile(r"how (?:do|will) they (?:mend|repair themselves) when\b", re.IGNORECASE)
REPAIR_EXCHANGE = re.compile(r'''^"(?:can(?:'t)?|could(?:n't)?) (?:you|we) (mend|repair) them\?"''', re.IGNORECASE)
REPAIR_REFUSAL = re.compile(r'^"(No[.,;!] [^"]+)"', re.IGNORECASE)
LIST_COMPANIONS = re.compile(
    r"what else (?:did (?:he|she|they) (?:say|report) )?([a-z]+) (?:along )?with them\W*", re.IGNORECASE,
)
COMPOUND_LIST = re.compile(r"\bof ((?:[a-z]+-and-)+[a-z]+)$", re.IGNORECASE)
ABOUT_SUBJECT = re.compile(rf'(?i:what about) ({NAME})\W*')
RECIPIENT = re.compile(rf'\b(?i:give|gave|offer|offered|lend|lent|send|sent|show|showed|bring|brought) ({NAME})(?= )')
NAME_ANSWER = re.compile(r"(?:(?i:the) )?[A-Z][a-z][\w'-]*(?: [A-Z][a-z][\w'-]*){0,3}")
PAST_ACTION = re.compile(rf'(?i:what did) (?P<actor>{NAME}) (?P<verb>[a-z]+)(?P<rest> .*)?')
PAST_PERFECT_ACTION = re.compile(rf'(?i:what had) ({NAME}) ([a-z]+)(?: .*)?')
OBJECT_QUESTION = re.compile(r'what (?:[a-z-]+ ){0,3}(?:do|did|does|had|is|are|was|were|can|could)\b', re.IGNORECASE)
PERSONAL_REFERENCE = re.compile(r'\b(?:he|she|him|her|his|hers)\b', re.IGNORECASE)
DIRECTION_QUESTION = re.compile(r'\b(?:go|went|come|came|walk(?:ed)?|climb(?:ed)?|slide|slid) (up|down)\b', re.IGNORECASE)
DIRECTION_WORD = re.compile(r'\b(up|down)(?:wards?)?\b', re.IGNORECASE)
CHOICE_SUBJECT = re.compile(
    r"which ((?!of\b)[\w'-]+(?: [\w'-]+){0,3}?) ((?:was|were|went|fell|came|had|got|[\w-]{3,}ed)\b.*)",
    re.IGNORECASE,
)
PLURAL_OBJECT = re.compile(r"\b(?i:the|these|those) ((?:[a-z][\w'-]* ){0,4}[a-z][\w'-]*s)$")
RETURN_QUESTION = re.compile(rf'(?i:where (?:did|does)) ({NAME}) (?i:return)')
ITEM_LOCATION = re.compile(r'where (?:is|was) (?:it|that)(?: (really|actually))?\W*', re.IGNORECASE)
MISSING_EVENT = re.compile(r'went missing|(?:was|were) (?:lost|missing)|got lost|disappeared', re.IGNORECASE)
PRETENDED_SOURCE = re.compile(r' to (take|pull|get|bring)\b.*\b(?:out of|from) .+', re.IGNORECASE)
RESCUE_QUESTION = re.compile(rf'what (rescued|saved|freed) ({NAME})', re.IGNORECASE)
TRAVEL_QUESTION = re.compile(rf'(?i:how did) ({NAME}) (?i:reach|get to|arrive in) (.+)')
MEANS_ANSWER = re.compile(r"(?i:a|an|the) [\w'-]+(?: [\w'-]+){0,5}|[A-Z][\w'-]+(?: [A-Z][\w'-]+){1,4}")
ACTION_ACTOR = re.compile(r'who ([a-z]+)( .+)?', re.IGNORECASE)
NAMED_OBJECT = re.compile(rf' ({NAME})(?= |$)')
LOCATION_TRANSFER = re.compile(r'where did (he|she) (bring|carry|drive|lead|send|take) (him|her)\?$', re.IGNORECASE)
TRANSFER_PARTICIPLE = {'bring': 'brought', 'carry': 'carried', 'drive': 'driven',
                       'lead': 'led', 'send': 'sent', 'take': 'taken'}
REPEATED_ACTION_REASON = re.compile(r'why did (he|she) ([a-z]+) (?:it|that)\W*', re.IGNORECASE)
ACTION_REASON = re.compile(r'why (?:was|were) (?:he|she|they) ([a-z]+)ing\W*', re.IGNORECASE)
REACTION = re.compile(r'how did (he|she|they) (react|respond)\W*', re.IGNORECASE)
REPORTED_SPEECH = re.compile(r'\b(?:say|said|tell|told|warn|warned|promise|promised|threaten|threatened)\b', re.IGNORECASE)
REPEATED_PURPOSE = re.compile(r'why did (?:he|she|they) do (?:it|that)\W*', re.IGNORECASE)
INSTRUMENT = re.compile(r' with ((?:a|an|the) .+?)(?: (?:after|before|when|while) .+)?$', re.IGNORECASE)
ASK_INSTRUMENT = re.compile(
    r'what (?:did|does|do|would|could|will|can) .+\bwith(?: (?:when|after|before|while|as) .+)?\?*',
    re.IGNORECASE,
)
WITH_INSTRUMENT = re.compile(r'\bwith (?:(?:a|an|the|his|her|their) )?$', re.IGNORECASE)
INSTRUMENT_DETAIL = re.compile(r" (?:on|in|of|from) (?:a|an|the) [\w'-]+(?=[\"',.!?;:])", re.IGNORECASE)
COLLECTIVE = re.compile(r'(?:a|the) (?:(?:great|large|small|whole) )?(band|cloud|crowd|flock|herd|swarm)', re.IGNORECASE)
GROUP_OBSERVATION = re.compile(
    rf'(?i:what did) ({NAME}) (?i:see) ((?i:coming|approaching|flying|following|rushing|running)\b[^?]*?)'
    r'(?: (?i:when|as) [^?]+)?\?*',
)
IN_COLLECTIVE = re.compile(r'\b(?:they|them)\b.*\bin $', re.IGNORECASE)
SYMBOL_MEANING = re.compile(r'what (did|does) it (represent|stand for|symbolize)\W*', re.IGNORECASE)
LOST_ITEMS = re.compile(rf'(?i:where did) ({NAME}) (?i:lose them)\W*')
PAST_TENSE = {
    'bring': 'brought', 'buy': 'bought', 'catch': 'caught', 'come': 'came', 'do': 'did', 'drink': 'drank',
    'eat': 'ate', 'fall': 'fell', 'find': 'found', 'fly': 'flew', 'get': 'got', 'give': 'gave', 'go': 'went',
    'hide': 'hid', 'hold': 'held', 'leave': 'left', 'make': 'made', 'meet': 'met', 'put': 'put', 'run': 'ran',
    'say': 'said', 'see': 'saw', 'set': 'set', 'shoe': 'shod', 'speak': 'spoke', 'take': 'took', 'wear': 'wore',
}
BASE_VERB = {past: verb for verb, past in PAST_TENSE.items()}
AFTER_OBJECT = set('about after again along and around as at back before behind by down for from go in into leave '
                   'near off on or out over through to under up with'.split())
COMMON_PERSON_NOUNS = set('boy child father girl king man mother prince princess queen sorcerer wizard woman'.split())
SUBORDINATE_CLAUSE = re.compile(r'\b(?i:when|while|after|before|as|if|because|since|until|once)\b')
INFINITIVE = re.compile(r'\bto (?!(?:a|an|the|his|her|their|its)\b)[a-z]+\b')
IMPERSONAL_IT = re.compile(
    r'it (?:take|cost|(?:is|was) (?:so |very |too )?(?:easy|hard|difficult|possible|impossible))\b', re.IGNORECASE,
)
CAPITALIZED_WORD = re.compile(r"(?<=[\s(\"'])[A-Z][a-z]+(?:-[A-Z]?[a-z]+)*(?: [A-Z][a-z]+(?:-[A-Z]?[a-z]+)*)*")
PLACE_PREPOSITION = re.compile(r'\b(?:across|at|from|in|into|near|of|through|to|toward)\s+(?:the\s+)?$', re.IGNORECASE)
NON_REFERENT_LEAD = re.compile(
    r'\b(?:across|after|at|before|beside|by|during|from|in|inside|into|near|on|outside|over|since|through|'
    r'throughout|to|toward|under|until|'
    r'ask|asked|bring|brought|give|gave|lend|lent|offer|offered|promise|promised|send|sent|show|showed|'
    r'teach|taught|tell|told|warn|warned)\s+$',
    re.IGNORECASE,
)
NON_NOUN_WORD = r'am|and|are|did|does|had|has|her|his|is|its|or|the|their|was|were'
PLURAL_PHRASE = (
    rf"\b(?i:the) (?:(?!(?:{NON_NOUN_WORD})\b)[\w-]+ ){{0,3}}"
    rf"(?:(?!(?:{NON_NOUN_WORD}|as|this|thus)\b)[\w-]*[^\Ws]s|children|people|men|women)\b"
    rf"|(?:{NAME}) (?i:and) (?:{NAME})"
)
PLURAL_GROUP = re.compile(PLURAL_PHRASE)
PLURAL_AT_END = re.compile(rf'(?:{PLURAL_PHRASE})$')
DEFINITE_NOUN = re.compile(r'\b(?i:the) ([\w-]+)')
STOP_WORDS = set(
    'a an and are as at be been but by can could did do does for from had has have he her him his '
    'how i in is it its me of on or she should that the their them then there these they this those '
    'to too us was were what when where which who whom why will with would you your'.split()
)
SILENT_E_VERBS = 'arrive believe escape live move name notice raise release rescue save serve'.split()
VERB_VARIANTS = {
    variant: verb
    for verb in SILENT_E_VERBS
    for variant in (verb + 'd', verb + 's', verb[:-1] + 'ing')
}


def search_terms(text):
    for word in re.findall(r'[a-z0-9]+', text.casefold()):
        if len(word) < 2 or word in STOP_WORDS:
            continue
        if word in VERB_VARIANTS:
            yield VERB_VARIANTS[word]
            continue
        for suffix in ('ing', 'ed', 's'):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        yield word


def _past_tense(verb):
    return PAST_TENSE.get(verb, re.sub(r'([^aeiou])y$', r'\1i', verb) + ('d' if verb.endswith('e') else 'ed'))


def _possessive(question, match):
    pronoun = match[0].lower()
    if pronoun != 'her':
        return pronoun in {'his', 'hers', 'its', 'their'}
    following = re.match(r'\s+([a-z]+)\b', question[match.end():], re.IGNORECASE)
    return following is not None and following[1].lower() not in AFTER_OBJECT


def _referent_scope(before, pronoun, possessive):
    """Return text that can hold a referent and whether that referent must directly precede "to"."""
    if possessive and pronoun != 'its':
        return before, False
    # Nouns in a time clause set the scene. When unsure, keep the conversation's referent.
    clause = SUBORDINATE_CLAUSE.search(before)
    if clause:
        return before[:clause.start()], False
    if pronoun in {'he', 'she', 'they', 'its'}:
        return before, False
    # An object pronoun normally cannot refer to the subject of its own clause.
    infinitive = max((match.start() for match in INFINITIVE.finditer(before)
                      if pronoun in {'them', 'it'} or CAPITALIZED_WORD.search(before, match.end())), default=-1)
    if infinitive < 0:
        return '', False
    return before[:infinitive].rstrip(), pronoun in {'them', 'it'}


def _can_name_person(text, pronoun):
    pattern = FEMALE_NAME if pronoun in {'she', 'her', 'hers'} else MALE_NAME
    if any(pattern.fullmatch(match[0]) for match in BOOK_NAME.finditer(text)):
        return True
    unlisted = BOOK_NAME.sub(' ', text)
    for match in CAPITALIZED_WORD.finditer(unlisted):
        plural = match[0].endswith('s') and not match[0].endswith('ss')
        if not plural and not PLACE_PREPOSITION.search(unlisted[:match.start()]):
            return True
    return False


def _can_name_group(text, at_end):
    pattern = PLURAL_AT_END if at_end else PLURAL_GROUP
    # A scene or listener, such as "at the hearings" or "tell the guards", is not a reliable referent.
    return any(not NON_REFERENT_LEAD.search(text[:match.start()]) for match in pattern.finditer(text))


def _can_name_thing(text, pronoun, at_end):
    # Plain "it" is ambiguous unless it is the object of a purpose clause: "hide the lantern to keep it dry".
    if pronoun == 'it' and not at_end:
        return False
    matches = list(DEFINITE_NOUN.finditer(text))
    if at_end:
        matches = [match for match in matches[-1:] if len(text[match.end(1):].split()) <= 2]
    for match in matches:
        if NON_REFERENT_LEAD.search(text[:match.start()]):
            continue
        word = match[1]
        if word == 'Sawhorse' or (word.islower() and word not in COMMON_PERSON_NOUNS):
            return True
        if pronoun == 'its' and word[0].isupper() and not BOOK_NAME.fullmatch(word):
            return True
    return False


def _has_antecedent(question, match):
    """Return whether an earlier word in the same question can be this pronoun's referent."""
    pronoun = match[0].lower()
    if pronoun == 'it' and IMPERSONAL_IT.match(question, match.start()):
        return True
    before, at_end = _referent_scope(question[:match.start()], pronoun, _possessive(question, match))
    if pronoun in {'they', 'them', 'their'}:
        return _can_name_group(before, at_end)
    if pronoun in {'it', 'its'}:
        return _can_name_thing(before, pronoun, at_end)
    return _can_name_person(before, pronoun)


def uses_book_history(question):
    if SHORT_FOLLOW_UP.search(question) or ABOUT_SUBJECT.fullmatch(question) or DAMAGE_REFERENCE.fullmatch(question):
        return True
    pronouns = list(PRONOUN.finditer(question))
    if not pronouns:
        return False
    main_question = question.rsplit(', ', 1)[-1].rstrip('?.!')
    if (all(_possessive(question, match) for match in pronouns)
            and (QUESTION_ACTOR.match(main_question) or BOOK_NAME.search(question[:pronouns[0].start()]))):
        return False
    return not all(_has_antecedent(question, match) for match in pronouns)


def _person_key(name):
    return name.casefold().removeprefix('the ')


def _person_referent(pronoun, actor, direct_object, people):
    name_pattern = FEMALE_NAME if pronoun in {'she', 'her', 'hers'} else MALE_NAME
    actor_key = _person_key(actor)
    object_key = _person_key(direct_object)
    # A named object can be a better antecedent than the actor, especially in a two-person question.
    if pronoun not in {'he', 'she'} and object_key != actor_key and name_pattern.fullmatch(object_key):
        return direct_object
    if actor and (not people or people.keys() == {actor_key} or name_pattern.fullmatch(actor_key)):
        return actor
    matching = [name for key, name in people.items() if name_pattern.fullmatch(key)]
    return matching[0] if len(matching) == 1 else ''


def followup_question(question, previous_question, previous_answer):
    previous_question = previous_question.replace('’', "'").rstrip('?.!')
    if DAMAGE_REFERENCE.fullmatch(question) and (damage := DAMAGE_CLAUSE.search(previous_question)):
        target, action = damage[1], damage[2].lower()
        return f'What caused {target} to {DAMAGE_VERBS.get(action, action)}?'
    speech = SPEECH_TOPIC.fullmatch(question)
    previous_speech = SPEECH_TOPIC.fullmatch(previous_question)
    if speech and previous_speech and speech['topic'].lower() in {'it', 'that'}:
        begin, end = speech.span('topic')
        return question[:begin] + previous_speech['topic'] + question[end:]
    named = NAMED_SUBJECT.fullmatch(previous_question)
    prior_action = ACTION_ACTOR.fullmatch(previous_question)
    actor = named[1] if named else ''
    name = previous_answer.strip(' .?!"')
    if not actor and re.match(r'who\b', previous_question, re.IGNORECASE) and NAME_ANSWER.fullmatch(name):
        actor = 'the ' + name[4:] if name.startswith('The ') else name
    subject = QUESTION_ACTOR.match(previous_question)
    if not actor and subject:
        actor = subject[1].removesuffix("'s")
    if actor and re.fullmatch(r'why (?:that|this) name\W*', question, re.IGNORECASE):
        return f'Why was {actor} given that name?'

    if about := ABOUT_SUBJECT.fullmatch(question):
        subject = QUESTION_ACTOR.match(previous_question)
        if subject and (recipient := RECIPIENT.search(previous_question, subject.end())):
            start, end = recipient.span(1)
            return previous_question[:start] + about[1] + previous_question[end:] + '?'
        prior_subject = actor if actor and actor in previous_question else ''
        if not prior_subject and (mentioned := BOOK_NAME.search(previous_question)):
            prior_subject = mentioned[0]
        if prior_subject:
            return re.sub(rf'\b{re.escape(prior_subject)}\b', lambda _: about[1], previous_question) + '?'

    if re.fullmatch(r'why\W*', question, re.IGNORECASE):
        if prior_action:
            verb, rest = prior_action.groups()
            if base := BASE_VERB.get(verb.lower()):
                return f"Why did {actor or 'someone'} {base}{rest or ''}?"
        if re.match(r'(?:what|where|when|how) (?:did|does|was|were|is|could)\b', previous_question, re.IGNORECASE):
            return 'Why ' + previous_question.split(' ', 1)[1] + '?'

    returning = RETURN_QUESTION.fullmatch(previous_question)
    if returning and re.fullmatch(r'who (?:went|came) (?:too|along)\W*', question, re.IGNORECASE):
        actor = returning[1]
        return f'Who went with {actor} when {actor} returned?'

    if re.fullmatch(r'who (?:used|operated|wielded) (?:it|that)\W*', question, re.IGNORECASE):
        rescue = RESCUE_QUESTION.fullmatch(previous_question)
        if rescue:
            verb, person = rescue[1].lower(), rescue[2]
            if MEANS_ANSWER.fullmatch(name):
                means = re.sub(r'^(A|An|The) ', lambda match: match[0].lower(), name)
                return f'Who used {means} to {verb[:-1]} {person}?'
            return f'Who {verb} {person}?'
        travel = TRAVEL_QUESTION.fullmatch(previous_question)
        if travel:
            return f'Who brought {travel[1]} to {travel[2]}?'

    reason = ACTION_REASON.fullmatch(question)
    if reason and prior_action:
        stem = reason[1].lower()
        if prior_action[1].lower() in {_past_tense(stem), _past_tense(stem + 'e')}:
            return f"Why was someone {stem}ing{prior_action[2] or ''}?"

    item = ''
    action = PAST_ACTION.fullmatch(previous_question)
    choice = CHOICE_SUBJECT.fullmatch(previous_question)
    location = ITEM_LOCATION.fullmatch(question)
    if action:
        actor, verb, rest = action.group('actor', 'verb', 'rest')
        verb = verb.lower()
        if location and location[1] and verb == 'pretend' and (source := PRETENDED_SOURCE.fullmatch(rest or '')):
            return f'Where did {actor} really {source[1].lower()} the item from?'
        if verb in {'show', 'reveal', 'display', 'produce'} and re.fullmatch(r'where was it\W*', question, re.IGNORECASE):
            return f'Where did {actor} produce the item from?'
        item = f'the item {actor} {_past_tense(verb)}{rest or ""}'
    elif perfect := PAST_PERFECT_ACTION.fullmatch(previous_question):
        actor, verb = perfect.groups()
        item = f'the thing {actor} {verb.lower()}'
    elif choice:
        if location and MISSING_EVENT.fullmatch(choice[2]):
            return f'Where did the {choice[1]} really end up?'
        item = f'the {choice[1]} that {choice[2]}'
    elif re.match(r'what\b', previous_question, re.IGNORECASE) and not OBJECT_QUESTION.match(previous_question):
        item = f'the thing that {previous_question[5:]}'

    plural = PLURAL_OBJECT.search(previous_question)
    plural_item = plural[1] if plural else ''
    if plural_item and (lost := LOST_ITEMS.fullmatch(question)):
        return f'Where did the {plural_item} go when {lost[1]} lost them?'
    if (item or actor) and re.fullmatch(r'what kind\W*', question, re.IGNORECASE):
        return f'What kind of thing is {item or actor}?'
    if plural_item and re.fullmatch(r'how many\W*', question, re.IGNORECASE):
        return f'How many {plural_item}?'

    reported_age = REPORTED_AGE.fullmatch(question)
    if reported_age and (speaker := REPORTING_ACTOR.search(previous_question)):
        group = reported_age[2] or ''
        verb = 'were' if group or reported_age[1].lower() == 'they' else 'was'
        return f'How old {verb} {speaker[1]}{group}?'

    pair = COORDINATED_ACTOR.match(previous_question, subject.end()) if subject else None
    plural_person = PLURAL_PERSON.search(question)
    if pair and plural_person:
        time_clause = TIME_CLAUSE.search(previous_question)
        # An object in a time clause is a weaker antecedent than the named pair who acted.
        if not plural or (time_clause and time_clause.start() < plural.start()) or plural_person[0].lower() == 'they':
            people_who_acted = f'{subject[1]} and {pair[1]}'
            if _possessive(question, plural_person):
                people_who_acted += "'s"
            start, end = plural_person.span()
            return question[:start] + people_who_acted + question[end:]

    people = {_person_key(match[0]): match[0] for match in BOOK_NAME.finditer(previous_question)}
    direct_object = ''
    if subject:
        verb = re.match(r' [a-z]+', previous_question[subject.end():], re.IGNORECASE)
        if verb and (match := NAMED_OBJECT.match(previous_question, subject.end() + verb.end())):
            direct_object = match[1]
    elif prior_action and prior_action[2]:
        remainder = prior_action[2].removeprefix(' up') if prior_action[2].startswith(' up ') else prior_action[2]
        if match := NAMED_OBJECT.match(remainder):
            direct_object = match[1]
            if not named and _person_key(actor) == _person_key(direct_object):
                actor = ''

    repeated = REPEATED_ACTION_REASON.fullmatch(question)
    if repeated and prior_action and prior_action[2]:
        pronoun, verb = (part.lower() for part in repeated.groups())
        if prior_action[1].lower() == _past_tense(verb):
            person = _person_referent(pronoun, actor, direct_object, people) or 'someone'
            return f'Why did {person} {verb}{prior_action[2]}?'

    if transfer := LOCATION_TRANSFER.fullmatch(question):
        subject_pronoun, verb, object_pronoun = transfer[1].lower(), transfer[2].lower(), transfer[3].lower()
        mover = _person_referent(subject_pronoun, actor, direct_object, people)
        passenger = _person_referent(object_pronoun, actor, direct_object, people)
        if passenger and _person_key(passenger) != _person_key(mover):
            # Ask for the destination so the reader can distinguish it from the starting place.
            if mover:
                return f'Where did {mover} {verb} {passenger} to?'
            return f'Where was {passenger} {TRANSFER_PARTICIPLE[verb]} to?'

    last_object = None

    def replace_pronoun(match):
        nonlocal last_object
        if _has_antecedent(question, match):
            return match[0]
        pronoun = match[0].lower()
        possessive = _possessive(question, match)
        if pronoun in {'they', 'them', 'their'}:
            referent = f'the {plural_item}' if plural_item else ''
        elif pronoun in {'it', 'its'}:
            referent = item or direct_object or actor
        else:
            referent = _person_referent(pronoun, actor, direct_object, people)
        if not referent:
            return match[0]
        if possessive and last_object:
            previous_match, previous_referent = last_object
            # "Show Zeb his room" keeps the immediate possessive grammatical and unambiguous.
            if referent == previous_referent and question[previous_match.end():match.start()].isspace():
                return match[0]
        if pronoun in {'him', 'her'} and not possessive:
            last_object = match, referent
        return f"{referent}'s" if possessive else referent

    expanded = PRONOUN.sub(replace_pronoun, question)
    if expanded != question:
        return expanded
    event = re.sub(r'^\s*(?:who|what|where|when|why|how)\b\s*', '', previous_question, flags=re.IGNORECASE)
    return f'{question.rstrip("?")} (context: {event})?'


@dataclass(frozen=True)
class Passage:
    section: str
    start_line: int
    end_line: int
    text: str
    heading: str = ''
    word_start: int = field(default=0, repr=False, compare=False)
    sentence_starts: tuple[int, ...] = field(default=(), repr=False, compare=False)
    paragraph_bounds: tuple[int, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class PassageEdge:
    text: str
    line: int
    paragraph_terms: frozenset[str]


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    margin: float
    context_end: int | None = None
    start: int | None = None


@dataclass(frozen=True)
class PassageReread:
    query: str
    passage: Passage
    span: AnswerSpan
    excerpt: str | None = None
    answer: str | None = None


@dataclass(frozen=True)
class PassageTrace:
    query: str
    previous_question: str
    previous_answer: str
    candidates: tuple[tuple[Passage, float], ...]
    spans: tuple[AnswerSpan, ...]
    rejected_query: str | None = None
    reread: PassageReread | None = None


@dataclass(frozen=True)
class BookReply:
    text: str
    source: Passage | None
    history_used: bool
    passage_trace: PassageTrace | None = None


def book_passages(text):
    for passage, _, _ in _passage_windows(text):
        yield passage


def _passage_windows(text):
    lines = text.splitlines()
    chapters = []
    for index, line in enumerate(lines):
        match = CHAPTER_HEADING.fullmatch(line.strip())
        if match:
            chapters.append((index, f'Chapter {match[1]}'))

    front_end = chapters[0][0] if chapters else len(lines)
    contents = next((i for i, line in enumerate(lines[:front_end]) if line.strip() == 'LIST OF CHAPTERS'), front_end)
    preface = next((i for i, line in enumerate(lines[:contents]) if line.strip() == 'To My Readers'), contents)
    sections = [(0, preface, 'Title page', ''), (preface, contents, 'Preface', '')]
    for (start, label), (end, _) in zip(chapters, chapters[1:] + [(len(lines), '')]):
        heading = next((line.strip() for line in lines[start + 1:end] if line.strip()), '')
        sections.append((start, end, label, heading if heading.isupper() else ''))

    for start, end, label, heading in sections:
        words = [(word, number + 1) for number in range(start, end) for word in lines[number].split()]
        sentences = [0]
        paragraphs = [0]
        for position, (word, line) in enumerate(words[:-1], start=1):
            next_word, next_line = words[position]
            if next_line > line + 1:
                paragraphs.append(position)
                sentences.append(position)
            elif (SENTENCE_END.search(word) and not ABBREVIATION.fullmatch(word)
                  and not next_word.lstrip("\"'`_“‘([")[:1].islower()):
                sentences.append(position)
        sentences.append(len(words))
        paragraphs.append(len(words))
        for offset in range(0, len(words), PASSAGE_STEP):
            limit = min(offset + PASSAGE_WORDS, len(words))
            window = words[offset:limit]
            first_index = bisect_right(sentences, offset) - 1
            last_index = bisect_left(sentences, limit)
            first, last = sentences[first_index], sentences[last_index]
            passage = Passage(
                label, window[0][1], window[-1][1], ' '.join(word for word, _ in window), heading,
                word_start=offset, sentence_starts=tuple(sentences[first_index:last_index]),
                paragraph_bounds=tuple(paragraphs[bisect_left(paragraphs, offset):bisect_right(paragraphs, limit)]),
            )
            prefix = suffix = None
            if 0 < offset - first <= PASSAGE_CONTEXT_WORDS:
                paragraph_end = min(paragraphs[bisect_right(paragraphs, offset)], limit)
                prefix = PassageEdge(
                    ' '.join(word for word, _ in words[first:offset]),
                    words[first][1],
                    frozenset(search_terms(' '.join(word for word, _ in words[offset:paragraph_end]))),
                )
            if 0 < last - limit <= PASSAGE_CONTEXT_WORDS:
                paragraph_start = max(offset, paragraphs[bisect_left(paragraphs, limit) - 1])
                suffix = PassageEdge(
                    ' '.join(word for word, _ in words[limit:last]),
                    words[last - 1][1],
                    frozenset(search_terms(' '.join(word for word, _ in words[paragraph_start:limit]))),
                )
            yield passage, prefix, suffix
            if limit == len(words):
                break


class BookIndex:
    def __init__(self, passages, *, passage_edges=None):
        self.passages = list(passages)
        if not self.passages:
            raise ValueError('The book contains no searchable passages.')
        self.passage_edges = passage_edges or {}
        self.frequencies = [Counter(search_terms(passage.text)) for passage in self.passages]
        self.span_lengths = [sum(terms.values()) for terms in self.frequencies]
        self.heading_terms = {
            passage.section: set(search_terms(passage.heading)) for passage in self.passages if passage.heading
        }
        self.section_terms = {section: set(terms) for section, terms in self.heading_terms.items()}
        for passage, terms in zip(self.passages, self.frequencies):
            self.section_terms.setdefault(passage.section, set()).update(terms)
        self.average_length = max(sum(self.span_lengths) / len(self.span_lengths), 1)
        document_counts = Counter(word for terms in self.frequencies for word in terms)
        total = len(self.passages)
        self.weights = {
            word: log1p((total - count + 0.5) / (count + 0.5))
            for word, count in document_counts.items()
        }

    @classmethod
    def from_text(cls, text):
        passages, edges = [], {}
        for passage, prefix, suffix in _passage_windows(text):
            passages.append(passage)
            if prefix or suffix:
                edges[passage] = (prefix, suffix)
        return cls(passages, passage_edges=edges)

    def _with_sentence_context(self, passage, question_terms):
        prefix, suffix = self.passage_edges.get(passage, (None, None))
        # Keep extra sentence text where the surrounding paragraph matches the question.
        prefix = prefix if prefix and prefix.paragraph_terms & question_terms else None
        suffix = suffix if suffix and suffix.paragraph_terms & question_terms else None
        if not (prefix or suffix):
            return passage
        text = ' '.join(part for part in (prefix.text if prefix else '', passage.text, suffix.text if suffix else '') if part)
        return replace(
            passage,
            start_line=prefix.line if prefix else passage.start_line,
            end_line=suffix.line if suffix else passage.end_line,
            text=text,
            word_start=passage.word_start - (len(prefix.text.split()) if prefix else 0),
        )

    def search(self, question, previous_question='', *, limit=SEARCH_LIMIT):
        query = dict.fromkeys(search_terms(question), 1.0)
        question_terms = set(query)
        for word in dict.fromkeys(search_terms(previous_question)):
            query[word] = query.get(word, 0.0) + 0.6

        heading_scores = {
            section: CHAPTER_HEADING_WEIGHT * sum(
                weight * self.weights.get(word, 0.0) for word, weight in query.items() if word in terms
            )
            for section, terms in self.heading_terms.items()
        }
        scores = []
        for passage, terms, length in zip(self.passages, self.frequencies, self.span_lengths):
            adjustment = 1.2 * (0.25 + 0.75 * length / self.average_length)
            score = sum(
                weight * self.weights[word] * (2.2 * terms[word]) / (terms[word] + adjustment)
                for word, weight in query.items() if word in terms
            )
            if score > 0:
                scores.append((passage, score + heading_scores.get(passage.section, 0.0)))
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:limit]
        return [(self._with_sentence_context(passage, question_terms), score) for passage, score in ranked]


class ExtractiveReader:
    def __init__(self):
        try:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Install the optional book lookup dependency with pip install "transformers>=4.45,<6".') from error
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL, revision=READER_REVISION, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            READER_MODEL, revision=READER_REVISION, use_safetensors=True,
        ).eval()

    @torch.inference_mode()
    def read(self, question, passages):
        question_ids = self.tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=80)
        question = self.tokenizer.decode(question_ids)
        batch = self.tokenizer(
            [question] * len(passages), [passage.text for passage in passages],
            max_length=384, padding=True, truncation='only_second',
            return_offsets_mapping=True, return_tensors='pt',
        )
        offsets = batch.pop('offset_mapping')
        result = self.model(**batch)
        answers = []
        for row, passage in enumerate(passages):
            context = batch.sequence_ids(row)
            first = context.index(1)
            last = len(context) - context[::-1].index(1)
            positions = torch.arange(last - first)
            lengths = positions[None, :] - positions[:, None]
            start = result.start_logits[row]
            end = result.end_logits[row]
            scores = start[first:last, None] + end[None, first:last]
            scores.masked_fill_((lengths < 0) | (lengths >= 24), float('-inf'))
            score, best = scores.flatten().max(dim=0)
            i, j = divmod(best.item(), last - first)
            begin = offsets[row, first + i, 0].item()
            finish = offsets[row, first + j, 1].item()
            margin = (score - start[0] - end[0]).item()
            context_end = offsets[row, last - 1, 1].item()
            answers.append(AnswerSpan(passage.text[begin:finish].strip(), margin, context_end, start=begin))
        return answers


def _sentence_start(passage, span):
    if not passage.sentence_starts:
        return None
    start = passage.text.find(span.text) if span.start is None else span.start
    word = passage.word_start + passage.text.count(' ', 0, start)
    return passage.sentence_starts[bisect_right(passage.sentence_starts, word) - 1]


def _independent_evidence(passage, span, other_passage, other_span):
    if passage.section != other_passage.section:
        return True
    start = _sentence_start(passage, span)
    other_start = _sentence_start(other_passage, other_span)
    if start is not None and other_start is not None:
        return start != other_start
    return passage.end_line < other_passage.start_line or other_passage.end_line < passage.start_line


def _supports_direction(passage, span, direction):
    start = passage.text.find(span.text) if span.start is None else span.start
    end = start + len(span.text)
    visible = passage.text[:span.context_end]
    cues = DIRECTION_WORD.finditer(visible, max(0, start - 100), end + 100)
    nearest = min(cues, key=lambda match: max(start - match.end(), match.start() - end), default=None)
    return nearest is not None and nearest[1].lower() == direction


def _select_answer(candidates, spans, question=''):
    strongest_search_score = candidates[0][1]
    motion = DIRECTION_QUESTION.search(question)
    answers = {}
    for (passage, score), span in zip(candidates, spans):
        if not (span.margin >= MIN_ANSWER_MARGIN and span.text and span.text in passage.text[:span.context_end]):
            continue
        key = ' '.join(span.text.casefold().split()).strip(' .!?,')
        key = re.sub(r'^(?:a|an|the) ', '', key)
        if key:
            strength = span.margin + RETRIEVAL_SCORE_WEIGHT * score / strongest_search_score
            answers.setdefault(key, []).append((strength, span, passage))

    ranked = []
    for matches in answers.values():
        strength, span, passage = max(matches, key=lambda match: match[0])
        if motion and len(matches) > 1:
            # Comparable extracts can refer to different trips. Prefer the one near the requested direction.
            alternatives = [match for match in matches if strength - match[0] <= DIRECTION_SCORE_TOLERANCE
                            and _supports_direction(match[2], match[1], motion[1].lower())]
            if alternatives:
                strength, span, passage = max(alternatives, key=lambda match: match[0])
        agrees = any(_independent_evidence(passage, span, other_passage, other_span)
                     for _, other_span, other_passage in matches)
        if agrees:
            strength += ANSWER_AGREEMENT_BONUS
        elif span.margin < MIN_SINGLE_PASSAGE_MARGIN:
            continue
        ranked.append((strength, span.margin, span.text, passage))

    if not ranked:
        return NO_ANSWER, None
    strongest = max(match[0] for match in ranked)
    # Keep almost equal search scores from overriding the reader's preference.
    near_best = (match for match in ranked if strongest - match[0] <= ANSWER_SCORE_TOLERANCE)
    _, _, text, source = max(near_best, key=lambda match: (match[1], match[0]))
    return text, source


def _select_expanded_answer(question, candidates, spans, initial_count):
    question_terms = set(search_terms(question))
    eligible = candidates[:initial_count]
    eligible_spans = spans[:initial_count]
    initial_texts = [
        passage.text[:span.context_end].casefold()
        for (passage, _), span in zip(eligible, eligible_spans)
    ]
    for candidate, span in zip(candidates[initial_count:], spans[initial_count:]):
        passage, _ = candidate
        matched = len(question_terms.intersection(search_terms(passage.text[:span.context_end])))
        enough_terms = matched >= EXPANDED_MATCH_FRACTION * len(question_terms)
        # Allow a specific phrase already mentioned in the original shortlist.
        repeated_phrase = (
            matched >= 2 and len(set(search_terms(span.text))) >= 2
            and any(span.text.casefold() in text for text in initial_texts)
        )
        if enough_terms or repeated_phrase:
            eligible.append(candidate)
            eligible_spans.append(span)
    return _select_answer(eligible, eligible_spans, question)


def _answer_sentence(passage, span, *, neighbors=0):
    if span.start is None or not passage.sentence_starts:
        return '', 0
    words = list(re.finditer(r'\S+', passage.text))
    starts = [words[boundary - passage.word_start].start() for boundary in passage.sentence_starts
              if 0 <= boundary - passage.word_start < len(words)]
    sentence = bisect_right(starts, span.start)
    if not sentence:
        return '', 0
    begin = starts[sentence - 1]
    end = starts[sentence] if sentence < len(starts) else len(passage.text)
    excerpt = passage.text[begin:end].rstrip()
    visible_end = len(passage.text) if span.context_end is None else span.context_end
    if not SENTENCE_END.search(excerpt) or not span.start + len(span.text) <= begin + len(excerpt) <= visible_end:
        return '', 0
    if neighbors:
        begin = starts[max(0, sentence - 1 - neighbors)]
        neighbor_end = sentence + neighbors
        wider_end = starts[neighbor_end] if neighbor_end < len(starts) else len(passage.text)
        wider = passage.text[begin:wider_end].rstrip()
        if begin + len(wider) <= visible_end and SENTENCE_END.search(wider):
            return wider, begin
        return passage.text[begin:end].rstrip(), begin
    return excerpt, begin


def _complete_paragraphs(passage):
    offsets = [word.start() for word in re.finditer(r'\S+', passage.text)] + [len(passage.text)]
    boundaries = [offsets[start - passage.word_start] for start in passage.paragraph_bounds]
    return [passage.text[begin:end].rstrip() for begin, end in zip(boundaries, boundaries[1:])]


def _tagged_speech(paragraph):
    quotes = list(SPEECH_QUOTE.finditer(paragraph))
    if not 1 <= len(quotes) <= 2 or not paragraph.startswith('"') or paragraph.count('"') != 2 * len(quotes):
        return '', ''
    tag_end = quotes[1].start() if len(quotes) == 2 else len(paragraph)
    tag = SPEECH_TAG.match(paragraph[quotes[0].end():tag_end].lstrip(' ,;'))
    if not tag:
        return '', ''
    speech = paragraph[:quotes[-1].end()] if len(quotes) == 2 else quotes[0][0]
    return tag['after'] or tag['before'], speech


def _repeated_speech(question, previous_question, text, previous_answer):
    request = SPEECH_TOPIC.fullmatch(question)
    previous = SPEECH_TOPIC.fullmatch(previous_question)
    return (
        request and previous and request['topic'].lower() in {'it', 'that'}
        and _person_key(request['speaker']) != _person_key(previous['speaker'])
        and text.strip(' ".,!?').casefold() == previous_answer.strip(' ".,!?').casefold()
    )


def _overlapping_passages(first, second):
    return (first.section == second.section and first.start_line <= second.end_line
            and second.start_line <= first.end_line)


def _instrument_answer(question, passage, span):
    if not ASK_INSTRUMENT.fullmatch(question) or not re.fullmatch(r"(?:(?i:a|an|the) )?[a-z][\w'-]*", span.text):
        return span.text
    if not WITH_INSTRUMENT.search(passage.text[:span.start]):
        return span.text
    end = span.start + len(span.text)
    detail = INSTRUMENT_DETAIL.match(passage.text[:span.context_end], end)
    return passage.text[span.start:detail.end()] if detail else span.text


class BookSession:
    def __init__(self, index, reader, *, trace_passages=False):
        self.index = index
        self.reader = reader
        self.trace_passages = trace_passages
        self.reset()

    @classmethod
    def from_book(cls, path=BOOK_PATH, *, trace_passages=False):
        book = path.read_bytes()
        index = BookIndex.from_text(book.decode('utf-8-sig'))
        session = cls(index, ExtractiveReader(), trace_passages=trace_passages)
        session.book_sha256 = hashlib.sha256(book).hexdigest()
        return session

    def reset(self):
        self.previous_question = ''
        self.previous_answer = ''
        self.previous_source = None

    def remember(self, question, answer, *, source=None):
        self._remember_resolved(self._followup(question) or question, answer, source)

    def _followup(self, question):
        if self.previous_question and uses_book_history(question):
            return followup_question(question, self.previous_question, self.previous_answer)
        return None

    def _remember_resolved(self, question, answer, source=None):
        # Avoid accumulating repeated context annotations across turns.
        resolved, annotation, _ = question.partition(' (context: ')
        self.previous_question = resolved + '?' if annotation else question
        self.previous_answer = answer
        self.previous_source = source

    def _read_question(self, reader_question, earlier_question):
        candidates = self.index.search(reader_question, earlier_question)
        text, source = NO_ANSWER, None
        spans = []
        if candidates:
            spans = self.reader.read(reader_question, [passage for passage, _ in candidates])
            text, source = _select_answer(candidates, spans, reader_question)
            weak_followup = earlier_question and any(
                passage == source and span.margin < FOLLOWUP_RECHECK_MARGIN
                for (passage, _), span in zip(candidates, spans)
            )
            if source is None or weak_followup:
                expanded = self.index.search(reader_question, earlier_question, limit=EXPANDED_SEARCH_LIMIT)
                initial_count = len(candidates)
                extra_passages = [passage for passage, _ in expanded[initial_count:]]
                if extra_passages:
                    spans += self.reader.read(reader_question, extra_passages)
                    candidates = expanded
                    text, source = _select_expanded_answer(reader_question, candidates, spans, initial_count)
        return text, source, candidates, spans

    def _reread_followup(self, question, text, source, candidates):
        previous = self.previous_source
        answer = self.previous_answer
        if previous is None:
            return None

        meaning = SYMBOL_MEANING.fullmatch(question)
        reaction = REACTION.fullmatch(question)
        promise = PROMISE.fullmatch(question)
        recipient = APPEARANCE.match(self.previous_question)
        reassurance = promise and recipient and source is None and DISTRESS.search(answer)
        companions = LIST_COMPANIONS.fullmatch(question)
        listed = COMPOUND_LIST.search(answer.strip(' ."!?'))
        if source is None and companions and listed:
            *others, last = listed[1].lower().split('-and-')
            group = ', '.join(others) + ' and ' + last
            query = f'What else {companions[1].lower()} with the {group}?'
        elif reassurance:
            query = f'What did {promise[1]} tell {recipient[1]} to reassure them?'
        elif meaning and MEANS_ANSWER.fullmatch(answer):
            referent = re.sub(r'^(?:a|an|the) ', '', answer, flags=re.IGNORECASE)
            query = f'What {meaning[1].lower()} the {referent} {meaning[2].lower()}?'
        elif len(set(search_terms(answer))) < 3:
            return None
        elif reaction and REPORTED_SPEECH.search(self.previous_question):
            if source is not None and _overlapping_passages(source, previous):
                return None
            person, verb = (part.lower() for part in reaction.groups())
            query = f'How did {person} {verb} when {person} heard "{answer}"?'
        elif REPEATED_PURPOSE.fullmatch(question) and answer.casefold() in text.casefold():
            action = PAST_ACTION.fullmatch(self.previous_question.rstrip('?.!'))
            instrument = INSTRUMENT.fullmatch(action['rest'] or '') if action else None
            if not instrument:
                return None
            query = f'What was the effect of using {instrument[1]}?'
        else:
            return None

        # The quoted words or action came from this scene. Recheck nearby evidence from the shortlist.
        local = [(passage, score) for passage, score in candidates[:SEARCH_LIMIT]
                 if _overlapping_passages(passage, previous)]
        if not local:
            return None
        checked = self.reader.read(query, [passage for passage, _ in local])
        candidates_and_spans = [(candidate, span) for candidate, span in zip(local, checked)
                                if span.text and span.text.casefold() not in answer.casefold()
                                and answer.casefold() not in span.text.casefold()]
        if reassurance:
            candidates_and_spans = [
                (candidate, span) for candidate, span in candidates_and_spans
                if any(span.text.strip('"') in speech and _person_key(person) == _person_key(promise[1])
                       for person, speech in map(_tagged_speech, _complete_paragraphs(candidate[0])))
            ]
        if not candidates_and_spans:
            return None
        eligible, spans = zip(*candidates_and_spans)
        _, source = _select_answer(eligible, spans, query)
        return next((PassageReread(query, passage, span) for (passage, _), span in candidates_and_spans
                     if passage == source), None)

    def _reread_repair(self, question, candidates):
        damage = DAMAGE_CLAUSE.search(question.rstrip('?.!'))
        if self.previous_source is None or not REPAIR_FOLLOWUP.match(question) or not damage:
            return None
        terms = set(search_terms(damage[1]))
        action = DAMAGE_VERBS.get(damage[2].lower(), damage[2].lower())
        for passage, _ in candidates[:SEARCH_LIMIT]:
            if not _overlapping_passages(passage, self.previous_source):
                continue
            paragraphs = _complete_paragraphs(passage)
            for incident, spoken_question, response in zip(paragraphs, paragraphs[1:], paragraphs[2:]):
                exchange = REPAIR_EXCHANGE.match(spoken_question)
                refusal = REPAIR_REFUSAL.match(response)
                scene_terms = set(search_terms(incident))
                if not (exchange and refusal and terms.issubset(scene_terms)
                        and scene_terms.intersection({action, damage[2].lower()})):
                    continue
                excerpt = ' '.join((incident, spoken_question, response))
                query = f'What will they do if the people cannot {exchange[1].lower()} them?'
                checked, = self.reader.read(query, [replace(passage, text=excerpt)])
                if checked.margin >= MIN_SINGLE_PASSAGE_MARGIN and checked.text and checked.text in refusal[1]:
                    answer = refusal[1].rstrip(',') if len(refusal[1].split()) <= 32 else None
                    return PassageReread(query, passage, checked, excerpt=excerpt, answer=answer)
        return None

    def _reread_collective(self, question, passage, span):
        collective = COLLECTIVE.fullmatch(span.text)
        observation = GROUP_OBSERVATION.fullmatch(question)
        if not collective or not observation:
            return None

        sentence, offset = _answer_sentence(passage, span)
        if not IN_COLLECTIVE.search(sentence[:span.start - offset]):
            return None

        query = f'Who made up the {collective[1].lower()} that {observation[1]} saw {observation[2]}?'
        checked, = self.reader.read(query, [passage])
        if (checked.margin < MIN_SINGLE_PASSAGE_MARGIN or checked.text not in sentence
                or not set(search_terms(checked.text))
                or checked.text.casefold() in span.text.casefold()
                or _person_key(checked.text) == _person_key(observation[1])
                or _answer_sentence(passage, checked) != (sentence, offset)):
            return None
        return PassageReread(query, passage, checked)

    def _reread_speech(self, question, candidates, previous=None):
        request = SPEECH_TOPIC.fullmatch(question)
        if not request:
            return None
        speaker, listener, topic = request.group('speaker', 'listener', 'topic')
        terms = set(search_terms(topic))
        if not terms or topic.lower() in {'it', 'that'}:
            return None

        confirmed = {}
        for passage, _ in candidates[:SEARCH_LIMIT]:
            if previous is not None and not _overlapping_passages(passage, previous):
                continue
            paragraphs = _complete_paragraphs(passage)
            for i, paragraph in enumerate(paragraphs):
                attributed_to, speech = _tagged_speech(paragraph)
                if not speech or speech in confirmed:
                    continue
                excerpt, query = paragraph, question
                if listener:
                    # In a dialogue, the previous speaker can ask the named person directly.
                    lead_in = paragraphs[i - 1] if i else ''
                    asking, spoken_question = _tagged_speech(lead_in)
                    addressed = re.search(
                        rf'\b(?:asking|appealing to|turning to) {re.escape(speaker)}\b', lead_in, re.IGNORECASE,
                    )
                    if (_person_key(asking) != _person_key(listener) or '?' not in spoken_question
                            or not terms.issubset(search_terms(spoken_question))):
                        continue
                    if _person_key(attributed_to) != _person_key(speaker):
                        if not addressed or attributed_to != SPEECH_ALIASES.get(_person_key(speaker)):
                            continue
                        query = question.replace(speaker, attributed_to, 1)
                    excerpt = ' '.join(paragraphs[i - 1:i + 2])
                elif _person_key(attributed_to) != _person_key(speaker) or not terms.issubset(search_terms(speech)):
                    continue
                checked, = self.reader.read(query, [replace(passage, text=excerpt)])
                extract = checked.text.strip(' "')
                if (checked.margin >= MIN_SINGLE_PASSAGE_MARGIN and extract
                        and any(extract in quote for quote in SPEECH_QUOTE.findall(speech))):
                    # Include both parts of a split quotation when they share the same attribution.
                    answer = speech if speech.count('"') > 2 else speech.strip('"').rstrip(',')
                    confirmed[speech] = PassageReread(query, passage, checked, excerpt=excerpt, answer=answer)
        return next(iter(confirmed.values())) if len(confirmed) == 1 else None

    def _reread_group_request(self, question, candidates):
        request = GROUP_REQUEST.fullmatch(question)
        if not request:
            return None
        speaker, setting = request.groups()
        place = re.findall(r'\b[A-Z][a-z-]+\b', setting)
        if not place:
            return None
        query = f'What did {speaker} suggest they should do?'
        confirmed = {}
        for passage, _ in candidates:
            if not set(search_terms(place[-1])).intersection(search_terms(passage.heading)):
                continue
            paragraphs = _complete_paragraphs(passage)
            for narration, speech in zip(paragraphs, paragraphs[1:]):
                if not re.match(rf'{re.escape(speaker)}\b', narration, re.IGNORECASE):
                    continue
                quote = re.match(r'^"(Let\'s [^"]+)" (?:he|she) said\b', speech)
                if not quote:
                    continue
                excerpt = f'{narration} {speech}'
                checked, = self.reader.read(query, [replace(passage, text=excerpt)])
                if checked.margin >= MIN_SINGLE_PASSAGE_MARGIN and checked.text and checked.text in quote[1]:
                    confirmed[quote[1]] = PassageReread(query, passage, checked, excerpt=excerpt)
        return next(iter(confirmed.values())) if len(confirmed) == 1 else None

    def _reread_request_reason(self, speaker, candidates):
        previous = self.previous_source
        action = re.sub(r"^let's ", '', self.previous_answer.rstrip('!.?'), flags=re.IGNORECASE)
        query = f'Why did {speaker} want them to {action}?'
        confirmed = []
        for passage, _ in candidates:
            if not _overlapping_passages(passage, previous):
                continue
            paragraphs = _complete_paragraphs(passage)
            for i, paragraph in enumerate(paragraphs):
                attributed_to, speech = _tagged_speech(paragraph)
                if _person_key(attributed_to) != _person_key(speaker):
                    continue
                excerpt = ' '.join(paragraphs[max(0, i - 2):i + 1])
                checked, = self.reader.read(query, [replace(passage, text=excerpt)])
                if checked.margin >= MIN_SINGLE_PASSAGE_MARGIN and checked.text and checked.text in speech:
                    reread = PassageReread(query, passage, checked, excerpt=excerpt)
                    confirmed.append((reread, paragraphs[max(0, i - 2):i]))
        if not confirmed:
            return None
        best, lead_in = max(confirmed, key=lambda result: result[0].span.margin)
        return self._expand_request_reason(speaker, best, lead_in)

    def _expand_request_reason(self, speaker, reason, lead_in):
        description = DESCRIPTION.fullmatch(reason.span.text)
        if not description:
            return reason
        speaker_key = _person_key(speaker)
        alias = SPEECH_ALIASES.get(speaker_key)
        query = f'Why did {speaker} think it was {description[1]}?'
        for paragraph in reversed(lead_in):
            attributed_to, speech = _tagged_speech(paragraph)
            if _person_key(attributed_to) != speaker_key and attributed_to != alias:
                continue
            checked, = self.reader.read(query, [replace(reason.passage, text=paragraph)])
            if checked.margin < MIN_SINGLE_PASSAGE_MARGIN or not checked.text:
                continue
            # Keep the longer explanation only when the reader confirms words in the same speaker's quote.
            detail = next((quote.strip('"').rstrip(',') for quote in SPEECH_QUOTE.findall(speech)
                           if checked.text in quote and 4 <= len(quote.split()) <= 40
                           and reason.span.text.casefold() not in quote.casefold()), None)
            if detail:
                answer = f'{reason.span.text.rstrip(".!?")}. {detail}'
                return PassageReread(query, reason.passage, checked, excerpt=paragraph, answer=answer)
        return reason

    def _reread_sentence(self, question, passage, span):
        excerpt, offset = _answer_sentence(passage, span)
        terms = set(search_terms(question))
        overlap = len(terms.intersection(search_terms(excerpt)))
        minimum = max(2, len(terms) / 2)
        age = (question.casefold().startswith('how old ') and span.margin >= MIN_ANSWER_MARGIN
               and AGE_ANSWER.fullmatch(span.text))
        # An explicit age can be verified in a sentence that refers to the speaker by pronoun.
        visible = passage.text[:span.context_end]
        if overlap >= minimum or (age and excerpt and len(terms.intersection(search_terms(visible))) >= 2):
            confirmed = self._confirm_span(question, passage, span, excerpt, offset)
            if confirmed is not None:
                return confirmed
        if overlap >= 2 and span.margin >= MIN_ANSWER_MARGIN:
            wider, begin = _answer_sentence(passage, span, neighbors=1)
            if wider != excerpt and len(terms.intersection(search_terms(wider))) >= minimum:
                confirmed = self._confirm_span(question, passage, span, wider, begin)
                if confirmed is not None:
                    return confirmed

        color = COLOR_QUESTION.fullmatch(question)
        if color and span.start is not None and span.margin >= 0:
            verb = 'is' if color[1].lower() in {'is', 'are'} else 'was'
            query = f'What {verb} the color of {color[2]}?'
            return self._confirm_span(query, passage, span, visible, 0)
        return None

    def _confirm_span(self, question, passage, span, excerpt, offset):
        # Confirm a weak extract in its complete sentence without trusting a new, unrelated answer.
        checked, = self.reader.read(question, [replace(passage, text=excerpt)])
        if (checked.margin < MIN_SINGLE_PASSAGE_MARGIN or checked.start is None
                or not checked.text or checked.text not in excerpt[:checked.context_end]):
            return None
        begin = offset + checked.start
        if begin >= span.start + len(span.text) or span.start >= begin + len(checked.text):
            return None
        return PassageReread(question, passage, checked, excerpt=excerpt)

    def chapter_terms(self, passage):
        return self.index.section_terms.get(passage.section, set()) | set(search_terms(passage.text))

    def confirm_candidates(self, trace, accepts, *, source=None):
        # The local comparison accounts for silent-e variation such as nibble and nibbled.
        def terms(text):
            return {term.removesuffix('e') if len(term) > 4 else term for term in search_terms(text)}

        query_terms = terms(trace.query)
        minimum_overlap = max(2, len(query_terms) / 2) if source is not None else 2
        confirmed = []
        for (passage, _), span in zip(trace.candidates[:SEARCH_LIMIT], trace.spans[:SEARCH_LIMIT]):
            if (source is not None and source != passage) or span.margin < 0 or not span.text:
                continue
            if not accepts(passage, span.text):
                continue
            excerpt, offset = _answer_sentence(passage, span, neighbors=1)
            if len(query_terms.intersection(terms(excerpt))) < minimum_overlap:
                continue
            reread = self._confirm_span(trace.query, passage, span, excerpt, offset)
            if (reread is not None and reread.span.margin >= MIN_ROUTE_CONFIRMATION_MARGIN
                    and accepts(passage, reread.span.text)):
                confirmed.append(reread)
        return max(confirmed, key=lambda reread: reread.span.margin, default=None)

    def remember_reply(self, reply):
        self._remember_resolved(reply.passage_trace.query, reply.text, reply.source)

    def reply(self, question, *, remember=True, remember_abstentions=True, include_trace=False):
        question = question.strip()
        if not question:
            raise ValueError('Empty prompt — please enter text')
        rewritten = self._followup(question)
        history_used = rewritten is not None
        earlier_question = self.previous_question if history_used else ''
        earlier_answer = self.previous_answer if history_used else ''
        reader_question = rewritten or question
        text, source, candidates, spans = self._read_question(reader_question, earlier_question)
        rejected_query = None
        if (source is None and rewritten and earlier_answer and PERSONAL_REFERENCE.search(question)
                and len(set(search_terms(question))) >= MIN_RETRY_TERMS
                and rewritten != followup_question(question, earlier_question, '')):
            # Keep the earlier question as a search cue, without binding the pronoun to the extracted name.
            fallback = self._read_question(question, earlier_question)
            if fallback[1] is not None:
                text, source, candidates, spans = fallback
                rejected_query, reader_question = reader_question, question
        speech_echo = source is not None and _repeated_speech(question, earlier_question, text, earlier_answer)
        if speech_echo:
            text, source = NO_ANSWER, None
        reread = self._reread_followup(question, text, source, candidates) if history_used else None
        previous_request = GROUP_REQUEST.fullmatch(earlier_question)
        reason = previous_request and self.previous_source and REQUEST_REASON.fullmatch(question)
        if reread is None and reason and (source is None or not _overlapping_passages(source, self.previous_source)):
            # A short question about a quoted request cannot point to a different scene.
            text, source = NO_ANSWER, None
            reread = self._reread_request_reason(previous_request[1], candidates)
        elif reread is None and source is None and candidates and spans:
            reread = self._reread_repair(question, candidates) if history_used else None
            previous = self.previous_source if history_used else None
            if reread is None:
                reread = self._reread_speech(reader_question, candidates, previous)
            if reread is None:
                reread = self._reread_group_request(reader_question, candidates)
            if reread is None and not speech_echo:
                reread = self._reread_sentence(reader_question, candidates[0][0], spans[0])
        elif reread is None and source is not None:
            selected = next((span for (passage, _), span in zip(candidates, spans) if passage == source), None)
            if selected is not None and selected.start is not None:
                reread = self._reread_collective(question, source, selected)
                if reread is None:
                    text = _instrument_answer(question, source, selected)
        if reread is not None:
            text, source = reread.answer or reread.span.text, reread.passage
            if PROMISE.fullmatch(question):
                text = text.strip('"')
            if reread.excerpt is None and REACTION.fullmatch(question):
                sentence, _ = _answer_sentence(source, reread.span)
                if sentence and len(sentence.split()) <= 32:
                    text = sentence
        trace = None
        if self.trace_passages or include_trace:
            trace = PassageTrace(reader_question, earlier_question, earlier_answer, tuple(candidates), tuple(spans),
                                 rejected_query=rejected_query, reread=reread)
        if remember and (source is not None or remember_abstentions):
            self._remember_resolved(reader_question, text if source else '', source)
        return BookReply(text, source, history_used, trace)
