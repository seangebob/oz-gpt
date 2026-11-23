import unittest
from unittest.mock import Mock, patch

from model.book_lookup import (
    AnswerSpan, BookIndex, BookSession, NO_ANSWER, Passage, book_passages, followup_question, search_terms,
)
from model.chat_evaluation import generate_session_predictions, grounded_correctness, passage_review_counts


BOOK = '''DOROTHY AND THE WIZARD IN OZ
BY L. FRANK BAUM
LIST OF CHAPTERS
1 THE EARTHQUAKE
CHAPTER 1.
Dorothy traveled with a kitten named Eureka.
CHAPTER 2.
Jim is a cab-horse, and Zeb drives the buggy.
'''


class BookLookupTests(unittest.TestCase):
    def setUp(self):
        self.passages = list(book_passages(BOOK))
        self.index = BookIndex(self.passages)

    def test_index_preserves_source_lines_and_excludes_contents(self):
        self.assertEqual([passage.section for passage in self.passages], ['Title page', 'Chapter 1', 'Chapter 2'])
        self.assertEqual(self.passages[1].start_line, 5)
        self.assertEqual(self.passages[1].end_line, 6)
        self.assertNotIn('LIST OF CHAPTERS', ' '.join(passage.text for passage in self.passages))
        self.assertEqual(self.index.search('Who wrote the book by Baum?')[0][0].section, 'Title page')

    def test_chapter_heading_helps_rank_passages_later_in_the_chapter(self):
        unrelated = Passage('Chapter 1', 10, 12, 'Zeb stopped on the road.', 'THE EARTHQUAKE')
        relevant = Passage('Chapter 2', 30, 32, 'Zeb stopped on the road.', 'ZEB RETURNS TO THE RANCH')
        index = BookIndex([unrelated, relevant])
        self.assertEqual(index.search('Where did Zeb return?')[0][0], relevant)

    def test_search_matches_silent_e_verb_forms_and_keeps_unrelated_words_distinct(self):
        variants = ('rescue rescues rescued rescuing', 'live lives lived living', 'name names named naming')
        for words in variants:
            with self.subTest(words=words):
                self.assertEqual(set(search_terms(words)), {words.split()[0]})
        self.assertTrue({'rescue', 'live', 'name'}.isdisjoint(search_terms('rescuer lively namesake')))

        relevant = Passage('Chapter 1', 3, 4, 'Eureka hurried to the rescue.')
        unrelated = Passage('Chapter 2', 5, 6, 'A rescuer was passing.')
        self.assertEqual([passage for passage, _ in BookIndex([unrelated, relevant]).search('Who rescued her?')],
                         [relevant])

    @patch('model.book_lookup.PASSAGE_WORDS', 10)
    @patch('model.book_lookup.PASSAGE_STEP', 8)
    def test_retrieval_completes_a_matching_sentence_at_either_edge(self):
        book = '''CHAPTER 1.

The little piglet hid where nobody could find it
until it fell into a big vase.

CHAPTER 2.

The Wizard was born in Omaha.
'''
        plain_index = BookIndex(book_passages(book))
        index = BookIndex.from_text(book)

        for question, first_line in [('Where was the piglet?', 1), ('What was in the vase?', 3)]:
            with self.subTest(question=question):
                original, original_score = plain_index.search(question)[0]
                extended, score = index.search(question)[0]
                self.assertEqual(score, original_score)
                self.assertIn(original.text, extended.text)
                self.assertIn('The little piglet hid where nobody could find it until it fell into a big vase.', extended.text)
                self.assertEqual((extended.start_line, extended.end_line), (first_line, 4))
                self.assertEqual(extended.section, 'Chapter 1')

        unrelated = 'Which chapter?'
        self.assertEqual(index.search(unrelated), plain_index.search(unrelated))

    def test_session_abstains_without_a_supported_span(self):
        reader = Mock()
        reader.read.return_value = [AnswerSpan('Kansas', 100)]
        session = BookSession(self.index, reader)
        self.assertEqual(session.reply('Who was Napoleon?', remember=False).text, NO_ANSWER)
        reader.read.assert_not_called()
        self.assertIsNone(session.reply('Who drives the buggy?').source)
        reader.read.return_value = [AnswerSpan('Zeb', 0)]
        self.assertIsNone(session.reply('Who drives the buggy?').source)

    def test_abstention_can_leave_book_history_for_a_chat_fallback(self):
        reader = Mock(read=Mock(return_value=[AnswerSpan('cab-horse', 8)]))
        session = BookSession(self.index, reader)
        first = session.reply('Who is Jim?')
        self.assertIsNotNone(first.source)
        reader.read.return_value = [AnswerSpan('', 0)]

        second = session.reply('Who drives the buggy?', remember_abstentions=False)

        self.assertIsNone(second.source)
        self.assertEqual(session.previous_source, first.source)
        self.assertEqual(session.previous_question, 'Who is Jim?')
        self.assertEqual(session.previous_answer, 'cab-horse')

    def test_answer_agreement_breaks_close_ties_and_keeps_the_best_citation(self):
        passages = [
            Passage('Chapter 1', 1, 2, 'An owl answered during rehearsal.'),
            Passage('Chapter 1', 3, 4, 'The fox answered for the council.'),
            Passage('Chapter 1', 5, 6, 'When it convened, a fox answered for the council.'),
        ]
        index = Mock()
        index.search.return_value = [(passage, 2.0) for passage in passages]
        reader = Mock()
        reader.read.return_value = [AnswerSpan('An owl', 8.3), AnswerSpan('The fox', 8.2), AnswerSpan('a fox', 7.5)]
        session = BookSession(index, reader)

        reply = session.reply('Who answered for the council?')
        self.assertEqual(reply.text, 'The fox')
        self.assertEqual(reply.source, passages[1])
        reader.read.return_value[0] = AnswerSpan('An owl', 10)
        self.assertEqual(session.reply('Who answered for the council?').source, passages[0])

    def test_a_weak_single_passage_answer_requires_agreement(self):
        first = Passage('Chapter 1', 1, 2, 'The fox answered.')
        second = Passage('Chapter 1', 3, 4, 'At noon the fox answered.')
        index = Mock()
        reader = Mock()
        reader.read.return_value = [AnswerSpan('fox', 2.8), AnswerSpan('fox', 2.9)]
        session = BookSession(index, reader)

        index.search.return_value = [(first, 2.0), (first, 2.0)]
        self.assertEqual(session.reply('Who answered?').text, NO_ANSWER)
        index.search.return_value = [(first, 2.0), (second, 2.0)]
        self.assertEqual(session.reply('Who answered?').text, 'fox')

    def test_weak_extract_can_be_confirmed_in_its_matching_sentence(self):
        sentence = 'Dana listed a new spring-clock as her invention.'
        passage = Passage('Chapter 1', 1, 2, sentence + ' Ezra bought a watch.', sentence_starts=(0, 8))
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))
        reader = Mock()
        weak = AnswerSpan('spring-clock', 0.1, start=sentence.index('spring-clock'))
        question = 'What invention did Dana list?'
        session = BookSession(index, reader, trace_passages=True)

        reader.read.side_effect = [[weak], [AnswerSpan('a new spring-clock', 5, start=sentence.index('a new'))]]
        reply = session.reply(question)
        self.assertEqual((reply.text, reply.source), ('a new spring-clock', passage))
        self.assertEqual(reply.passage_trace.reread.excerpt, sentence)
        self.assertEqual(reply.passage_trace.spans[0], weak)

        reader.read.side_effect = [[weak], [AnswerSpan('Dana', 9, start=0)]]
        self.assertIsNone(session.reply(question).source)

        reader.read.side_effect = [[AnswerSpan('spring-clock', 0.1, len(sentence) - 1, weak.start)]]
        self.assertIsNone(session.reply(question).source)

    def test_weak_extract_can_be_confirmed_with_a_neighboring_sentence(self):
        sentences = ('Dark clouds had cleared earlier.', 'Clara and Danny sat in the buggy in the sunshine.',
                     'The bright light made them cover their faces with their hands.', 'Later the pair went home.')
        text = ' '.join(sentences)
        passage = Passage('Chapter 1', 1, 4, text, sentence_starts=tuple(
            sum(len(sentence.split()) for sentence in sentences[:i]) for i in range(len(sentences))))
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))
        reader = Mock()
        excerpt = ' '.join(sentences[1:])
        reader.read.side_effect = [
            [AnswerSpan('cover their faces', 2.1, start=text.index('cover'))],
            [AnswerSpan('cover their faces', 5.0, start=excerpt.index('cover'))],
        ]
        session = BookSession(index, reader, trace_passages=True)

        reply = session.reply('What did Clara and Danny do with their hands in the bright sunshine in the buggy?')

        self.assertEqual((reply.text, reply.source), ('cover their faces', passage))
        self.assertEqual(reply.passage_trace.reread.excerpt, excerpt)

    def test_weak_age_can_be_verified_when_the_evidence_uses_a_pronoun(self):
        setup = 'The dragonette looked at its siblings.'
        evidence = 'All of us were sixteen years old.'
        text = f'{setup} {evidence}'
        passage = Passage('Chapter 1', 1, 2, text, sentence_starts=(0, len(setup.split())))
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))
        reader = Mock()
        reader.read.side_effect = [
            [AnswerSpan('sixteen years old', 2.6, start=text.index('sixteen'))],
            [AnswerSpan('sixteen', 4.0, start=evidence.index('sixteen'))],
        ]
        session = BookSession(index, reader)

        reply = session.reply('How old were the dragonette and its siblings?')

        self.assertEqual((reply.text, reply.source), ('sixteen', passage))

    def test_speech_reread_keeps_both_quotes_from_the_same_speaker(self):
        book = """CHAPTER 1.

THE WORKSHOP

"The clock has stopped," said Dana. "It needs a spring."

"I can repair the clock," said Ezra.
"""
        index = BookIndex.from_text(book)
        reader = Mock()
        reader.read.side_effect = [
            [AnswerSpan('I can repair the clock', 0.5)],
            [AnswerSpan('It needs a spring', 6)],
        ]
        session = BookSession(index, reader, trace_passages=True)

        reply = session.reply('What did Dana say about the clock?')

        self.assertEqual(reply.text, '"The clock has stopped," said Dana. "It needs a spring."')
        self.assertIn(reply.text, reply.source.text)
        self.assertEqual(reply.passage_trace.reread.span.text, 'It needs a spring')

    def test_speech_reread_cannot_use_a_different_speakers_topic(self):
        book = """CHAPTER 1.

THE WORKSHOP

"The clock has stopped," said Ezra.

"We will start soon," said Dana.
"""
        reader = Mock()
        reader.read.return_value = [AnswerSpan('The clock has stopped', 0.5)]
        session = BookSession(BookIndex.from_text(book), reader)

        self.assertEqual(session.reply('What did Dana say about the clock?').text, NO_ANSWER)
        reader.read.assert_called_once()

    def test_dialogue_followup_reuses_the_explicit_topic(self):
        self.assertEqual(
            followup_question(
                'What did Ezra tell Dana about it?',
                'What did Dana say about the clock during the meeting?',
                'The clock had stopped.',
            ),
            'What did Ezra tell Dana about the clock?',
        )

    def test_dialogue_followup_does_not_echo_the_other_speaker(self):
        book = """CHAPTER 1.

THE WORKSHOP

"The clock stopped," said Dana.

"We'll find a spring," said Ezra.
"""
        index = BookIndex.from_text(book)
        session = BookSession(index, Mock(), trace_passages=True)
        session.reader.read.return_value = [AnswerSpan('The clock stopped', 8)]
        prior = session.reply('What did Dana say about the clock?')
        reply = session.reply('What did Ezra tell Dana about it?')

        self.assertEqual(prior.text, 'The clock stopped')
        self.assertEqual(reply.text, NO_ANSWER)
        self.assertEqual(reply.passage_trace.query, 'What did Ezra tell Dana about the clock?')

    def test_request_reason_rejects_a_confident_answer_from_another_scene(self):
        book = """CHAPTER 1.

THE WORKSHOP

Dana checked the tools.

"Let's find a new spring!" she said.

CHAPTER 2.

THE COUNCIL

Dana wanted to hear from the council.
"""
        index = BookIndex.from_text(book)
        first, distant = index.passages
        reader = Mock()
        reader.read.return_value = [AnswerSpan('hear from the council', 12.0), AnswerSpan('new spring', 0.0)]
        session = BookSession(index, reader, trace_passages=True)
        session.previous_question = 'What did Dana ask everyone to do when they reached the Workshop?'
        session.previous_answer = 'find a new spring'
        session.previous_source = first
        with patch.object(index, 'search', return_value=[(distant, 4.0), (first, 3.0)]):
            reply = session.reply('Why did she want that?')

        self.assertEqual(reply.text, NO_ANSWER)
        self.assertIsNone(reply.source)
        self.assertEqual(reply.passage_trace.candidates[0][0], distant)

    def test_reaction_reread_quotes_the_previous_reply_and_stays_in_the_scene(self):
        speech = 'The canal will flood the lower fields'
        reaction = 'They were scared and ran to the hill.'
        scene = Passage('Chapter 1', 1, 2, f'{speech}. {reaction}', sentence_starts=(0, len(speech.split())))
        elsewhere = Passage('Chapter 1', 10, 12, 'Later the children were amused.')
        index = Mock()
        index.search.side_effect = [[(scene, 2.0)], [(scene, 4.0), (elsewhere, 3.0)]]
        reader = Mock()
        reader.read.side_effect = [[AnswerSpan(speech, 8)], [AnswerSpan('scared', 0), AnswerSpan('amused', 9)],
                                   [AnswerSpan('scared', 5, start=scene.text.index('scared'))]]
        session = BookSession(index, reader, trace_passages=True)

        session.reply('What did Dana say to the children?')
        reply = session.reply('How did they react?')

        self.assertEqual((reply.text, reply.source), (reaction, scene))
        reader.read.assert_called_with(f'How did they react when they heard "{speech}"?', [scene])
        self.assertEqual(reply.passage_trace.reread.span.text, 'scared')
        session.reset()
        self.assertIsNone(session.previous_source)

    def test_purpose_reread_asks_for_the_effect_when_the_reader_repeats_the_action(self):
        action = 'all the long wooden panels'
        effect = 'the wood would dry faster'
        scene = Passage('Chapter 1', 1, 4, f'Dana brushed {action}. Then {effect}.')
        index = Mock(search=Mock(return_value=[(scene, 2.0)]))
        reader = Mock()
        reader.read.side_effect = [[AnswerSpan(action, 8)], [AnswerSpan(f'brushed {action}', 10)],
                                   [AnswerSpan(effect, 5)]]
        session = BookSession(index, reader)

        session.reply('What did Dana brush with the mixture after doing her own boots?')
        reply = session.reply('Why did she do that?')

        self.assertEqual((reply.text, reply.source), (effect, scene))
        reader.read.assert_called_with('What was the effect of using the mixture?', [scene])

    def test_instrument_answer_keeps_the_attached_detail_visible_to_the_reader(self):
        text = 'Clara said to steer with the oars on the raft, since the spare oars were broken.'
        passage = Passage('Chapter 1', 1, 2, text)
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))
        reader = Mock(read=Mock(return_value=[AnswerSpan('oars', 8, start=text.index('oars'))]))
        session = BookSession(index, reader)
        question = 'What did Clara say to steer with when they left?'

        reply = session.reply(question)
        self.assertEqual((reply.text, reply.source), ('oars on the raft', passage))

        reader.read.return_value = [AnswerSpan('oars', 8, context_end=text.index('raf'), start=text.index('oars'))]
        self.assertEqual(session.reply(question).text, 'oars')

    def test_collective_answer_names_members_from_the_same_sentence(self):
        sentence = 'Pale riders swept toward the shore, and Clara saw them coming in a great crowd.'
        text = sentence + ' People watched from the pier.'
        passage = Passage('Chapter 1', 1, 3, text, sentence_starts=(0, len(sentence.split())))
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))
        reader = Mock()
        session = BookSession(index, reader, trace_passages=True)
        collective = AnswerSpan('a great crowd', 8, start=text.index('a great crowd'))
        question = 'What did Clara see coming toward the shore when she came outside?'

        reader.read.side_effect = [[collective], [AnswerSpan('Pale riders', 7, start=0)]]
        reply = session.reply(question)
        self.assertEqual((reply.text, reply.source), ('Pale riders', passage))
        self.assertEqual(reply.passage_trace.reread.query, 'Who made up the crowd that Clara saw coming toward the shore?')

        reader.read.side_effect = [[collective], [AnswerSpan('People', 10, start=text.index('People'))]]
        self.assertEqual(session.reply(question).text, 'a great crowd')

    def test_symbol_followup_rereads_the_meaning_of_the_cited_object(self):
        passage = Passage('Chapter 1', 1, 4,
                          'On the banner was a silver bell. Red stood for the farms, and the bell for the town hall.')
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))
        reader = Mock()
        reader.read.side_effect = [[AnswerSpan('a silver bell', 8)], [AnswerSpan('the farms', 9)],
                                   [AnswerSpan('the town hall', 6)]]
        session = BookSession(index, reader)

        session.reply('What was painted on the banner?')
        reply = session.reply('What did it represent?')

        self.assertEqual((reply.text, reply.source), ('the town hall', passage))
        reader.read.assert_called_with('What did the silver bell represent?', [passage])

    @patch('model.book_lookup.PASSAGE_WORDS', 12)
    @patch('model.book_lookup.PASSAGE_STEP', 5)
    def test_overlapping_windows_vote_once_for_the_same_sentence(self):
        book = ('CHAPTER 1.\n\nAn old owl sat nearby. The fox answered for the council. '
                'Later another fox answered the same question. The crowd gathered.')
        first, overlap, independent, *_ = BookIndex.from_text(book).passages
        index = Mock()
        reader = Mock()
        session = BookSession(index, reader)
        index.search.return_value = [(first, 2.0), (overlap, 2.0)]
        reader.read.return_value = [AnswerSpan('fox', 2.8, start=p.text.index('fox')) for p in (first, overlap)]
        self.assertEqual(session.reply('Who answered?').text, NO_ANSWER)

        index.search.return_value.append((independent, 2.0))
        reader.read.return_value.append(AnswerSpan('fox', 2.8, start=independent.text.index('fox')))
        self.assertEqual(session.reply('Who answered?').text, 'fox')

    def test_wider_search_reads_extra_passages_and_checks_their_topic(self):
        initial = Passage('Chapter 1', 1, 2, 'Iris promised her ferret a silver bell.')
        actual = Passage('Chapter 2', 3, 4, 'Iris found the ferret wearing a silver bell.')
        unrelated = Passage('Chapter 3', 5, 6, 'Iris wore her own ribbon.')
        index = Mock()
        index.search.side_effect = [[(initial, 5.0)], [(initial, 5.0), (unrelated, 4.0), (actual, 3.0)]]
        reader = Mock()
        reader.read.side_effect = [
            [AnswerSpan('ferret', 0)],
            [AnswerSpan('ribbon', 12), AnswerSpan('silver bell', 4)],
        ]
        session = BookSession(index, reader, trace_passages=True)
        question = "What marked Iris's own ferret?"

        reply = session.reply(question)

        self.assertEqual(reply.text, 'silver bell')
        self.assertEqual(reply.source, actual)
        index.search.assert_called_with(question, '', limit=24)
        reader.read.assert_called_with(question, [unrelated, actual])
        self.assertEqual(len(reply.passage_trace.candidates), 3)
        self.assertEqual(reply.passage_trace.spans[1].text, 'ribbon')

    def test_wider_search_can_recheck_a_weak_followup(self):
        first = Passage('Chapter 1', 1, 2, 'Dana returned with a scout.')
        actual = Passage('Chapter 2', 3, 4, 'Dana went with a guide when she returned.')
        index = Mock()
        index.search.side_effect = [[(first, 5.0)], [(first, 5.0), (actual, 4.0)]]
        reader = Mock()
        reader.read.side_effect = [[AnswerSpan('a scout', 4.5)], [AnswerSpan('a guide', 8)]]
        session = BookSession(index, reader)
        session.remember('Where does Dana return?', '')

        reply = session.reply('Who went too?')

        self.assertTrue(reply.history_used)
        self.assertEqual(reply.source, actual)
        reader.read.assert_called_with('Who went with Dana when Dana returned?', [actual])

    def test_followup_retries_without_an_unsupported_inferred_name(self):
        passage = Passage('Chapter 1', 1, 3, 'The runner climbed twenty feet while the dogs chased her.')
        index = Mock()
        index.search.return_value = [(passage, 2.0)]
        reader = Mock()
        reader.read.side_effect = [[AnswerSpan('twenty feet', 0.1)], [AnswerSpan('twenty feet', 8.0)]]
        session = BookSession(index, reader, trace_passages=True)
        previous = 'Who escaped the dogs?'
        session.remember(previous, 'Dana')
        question = 'How high did she climb?'

        reply = session.reply(question)

        self.assertEqual(reply.source, passage)
        self.assertTrue(reply.history_used)
        self.assertEqual(reply.passage_trace.query, question)
        self.assertEqual(reply.passage_trace.rejected_query, 'How high did Dana climb?')
        index.search.assert_called_with(question, previous)

    def test_generic_pronoun_followup_does_not_retry_without_the_name(self):
        passage = Passage('Chapter 1', 1, 3, 'The runner took Alice to prison.')
        index, reader = Mock(), Mock()
        index.search.return_value = [(passage, 2.0)]
        reader.read.return_value = [AnswerSpan('prison', 0.1)]
        session = BookSession(index, reader)
        session.remember('Who captured the spy?', 'Dana')

        self.assertEqual(session.reply('Where did he take her?').text, NO_ANSWER)
        reader.read.assert_called_once()

    def test_same_answer_cites_the_trip_in_the_requested_direction(self):
        up = Passage('Chapter 1', 1, 2, 'Dana climbed up and Alice followed her.')
        down = Passage('Chapter 1', 3, 4, 'Dana climbed downward and Alice followed her.')
        index, reader = Mock(), Mock()
        index.search.return_value = [(up, 2.0), (down, 1.9)]
        reader.read.return_value = [AnswerSpan('Alice', 8), AnswerSpan('Alice', 7.9)]
        session = BookSession(index, reader)

        self.assertEqual(session.reply('Who went down with Dana?').source, down)
        self.assertEqual(session.reply('Who went up with Dana?').source, up)

    def test_trace_keeps_search_order_and_marks_the_selected_passage(self):
        index = Mock()
        index.search.return_value = [(self.passages[1], 3.0), (self.passages[2], 1.0)]
        reader = Mock()
        reader.read.return_value = [AnswerSpan('Dorothy', 3, 24), AnswerSpan('Zeb', 9)]
        session = BookSession(index, reader, trace_passages=True)
        turns = [(1, {'messages': [
            {'role': 'user', 'content': 'Who drives the buggy?'},
            {'role': 'assistant', 'content': 'Zeb.'},
        ]})]
        rows = generate_session_predictions(session, turns, 'generated')
        trace = rows[0]['passage_trace']
        self.assertEqual(trace['selected_rank'], 2)
        self.assertEqual([p['rank'] for p in trace['candidates']], [1, 2])
        self.assertEqual(trace['candidates'][0]['reader_text'], self.passages[1].text[:24])
        self.assertEqual(trace['candidates'][1]['reader_span'], 'Zeb')
        self.assertEqual(passage_review_counts(rows), (0, 0, 1))
        trace['evidence_ranks'] = [2]
        self.assertEqual(passage_review_counts(rows), (1, 0, 0))
        trace['evidence_ranks'] = []
        self.assertEqual(passage_review_counts(rows), (0, 1, 0))
        trace['evidence_ranks'] = [3]
        with self.assertRaisesRegex(ValueError, 'retrieved passage ranks'):
            passage_review_counts(rows)

    def test_followups_use_previous_subject_and_report_citations(self):
        def read(question, passages):
            if question == 'Who is Jim?':
                text = 'cab-horse'
            elif question == 'Who is with Jim?':
                text = 'Zeb'
            else:
                text = ''
            return [AnswerSpan(text, 8) for _ in passages]

        reader = Mock()
        reader.read.side_effect = read
        session = BookSession(self.index, reader, trace_passages=True)
        turns = [(1, {'messages': [
            {'role': 'user', 'content': 'Who is Jim?'},
            {'role': 'assistant', 'content': 'A cab-horse.'},
            {'role': 'user', 'content': 'Who is with him?'},
            {'role': 'assistant', 'content': 'Zeb.'},
        ]})]
        rows = generate_session_predictions(session, turns, 'generated')
        self.assertEqual(rows[1]['reply'], 'Zeb')
        self.assertTrue(rows[1]['history_used'])
        self.assertTrue(rows[1]['reference_match'])
        self.assertEqual(rows[1]['source']['section'], 'Chapter 2')
        self.assertIn('Zeb', rows[1]['source']['text'])
        self.assertIsNone(rows[1]['source_supports_reply'])
        self.assertEqual(rows[1]['passage_trace']['query'], 'Who is with Jim?')
        self.assertEqual(rows[1]['passage_trace']['previous_question'], 'Who is Jim?')

        rows[1]['source_supports_reply'] = False
        self.assertEqual(grounded_correctness(rows[1:]), '0/1')
        rows[1]['source_supports_reply'] = True
        self.assertEqual(grounded_correctness(rows[1:]), '1/1')
        session.reset()
        result = session.reply('Who is with him?')
        self.assertEqual(result.text, NO_ANSWER)
        self.assertFalse(result.history_used)

    def test_followup_search_anchors_the_item_to_the_previous_action(self):
        index = Mock()
        index.search.return_value = []
        session = BookSession(index, Mock(), trace_passages=True)
        previous = 'What did Clara reveal at the fair?'
        session.remember(previous, 'exceedingly thoughtful')
        reply = session.reply('Where was it?')
        index.search.assert_called_once_with('Where did Clara produce the item from?', previous)
        self.assertEqual(reply.passage_trace.previous_answer, 'exceedingly thoughtful')

    def test_resolved_subject_survives_multiple_followups(self):
        for history in ('generated', 'reference'):
            with self.subTest(history=history):
                session = BookSession(self.index, Mock(read=Mock(return_value=[])), trace_passages=True)
                turns = [('What did Jim pull?', 'The buggy.'), ('Where did he go?', 'The ranch.'),
                         ('Who went with him?', 'Zeb.')]
                conversation = {'messages': [
                    {'role': role, 'content': text}
                    for question, answer in turns
                    for role, text in [('user', question), ('assistant', answer)]
                ]}
                rows = generate_session_predictions(session, [(1, conversation)], history)
                trace = rows[2]['passage_trace']
                self.assertEqual(trace['previous_question'], 'Where did Jim go?')
                self.assertEqual(trace['query'], 'Who went with Jim?')

                session.reply('Who is Dorothy?')
                self.assertEqual(session.reply('Where did she go?').passage_trace.query, 'Where did Dorothy go?')

    def test_transfer_followup_keeps_the_actor_and_object_in_separate_roles(self):
        previous = 'Who took Eureka away after Ozma ordered her held for trial?'
        question = 'Where did he take her?'
        self.assertEqual(followup_question(question, previous, 'The Captain-General.'),
                         'Where did the Captain-General take Eureka to?')
        for answer in ('Ozma', 'Eureka', ''):
            with self.subTest(earlier_answer=answer):
                self.assertEqual(followup_question(question, previous, answer), 'Where was Eureka taken to?')

        self.assertEqual(followup_question('Where did she lead him?', 'Who led zeb away from Dorothy?', 'Ozma'),
                         'Where did Ozma lead zeb to?')

    def test_followup_object_and_possessive_keep_the_matching_participant(self):
        previous = 'Where did Jellia say Dorothy was when Zeb asked at the palace?'
        self.assertEqual(followup_question('Who was going to show him his room?', previous, 'Ozma'),
                         'Who was going to show Zeb his room?')

    def test_reason_followup_restores_the_action_even_without_the_actor(self):
        previous = 'Who suggested taking the Royal Princess off her bush?'
        question = 'Why did he suggest it?'
        for answer, person in [('The Wizard.', 'the Wizard'), ('', 'someone')]:
            with self.subTest(earlier_answer=answer):
                self.assertEqual(followup_question(question, previous, answer),
                                 f'Why did {person} suggest taking the Royal Princess off her bush?')

    def test_actor_and_object_are_available_to_the_next_followup(self):
        index = Mock(search=Mock(return_value=[]))
        session = BookSession(index, Mock(), trace_passages=True)
        session.remember('Who took Eureka away?', 'Captain-General')

        session.reply('Where did he take her?')
        third_turn = session.reply('Where did she see him?')

        self.assertEqual(third_turn.passage_trace.previous_question, 'Where did Captain-General take Eureka to?')
        self.assertEqual(third_turn.passage_trace.query, 'Where did Eureka see Captain-General?')

    def test_followup_rewrites_use_the_question_for_people_and_objects(self):
        cases = [
            ('What did he fix?', 'What did the Tin Woodman use at the palace?', 'a branch',
             'What did the Tin Woodman fix?'),
            ('Where was it?', 'Which little dog went missing?', 'a neighbor',
             'Where did the little dog really end up?'),
            ('Where is it?', 'Which red dog was lost?', 'a neighbor', 'Where did the red dog really end up?'),
            ('Where was it?', 'Which crate was delivered?', 'a neighbor', 'Where was the crate that was delivered?'),
            ('Why that name?', 'Who is in the stable?', 'Comet', 'Why was Comet given that name?'),
            ('Where did she lose them?', 'Which witch owned the silver shoes?', 'the Wicked Witch',
             'Where did she lose the silver shoes?'),
            ('What let them open their eyes again?', 'What did Dorothy and Zeb do when the suns grew bright?', '',
             'What let Dorothy and Zeb open their eyes again?'),
            ('How old did it say it and its siblings had just turned?', 'What did the dragonette say it ate?', 'fish',
             'How old were the dragonette and its siblings?'),
            ('What caused that damage in the first place?', 'How do they mend when the roofs crack?', '',
             'What caused the roofs to crack?'),
            ('Where did he put it?', 'Who picked up the Gargoyle by the gate?', 'Zeb',
             'Where did Zeb put the Gargoyle?'),
        ]
        for question, previous, answer, expected in cases:
            with self.subTest(question=question, previous=previous):
                self.assertEqual(followup_question(question, previous, answer), expected)

    def test_followups_keep_the_earlier_action_and_location(self):
        cases = [
            ('Who operated it?', 'What freed Clara?', 'A winch.', 'Who used a winch to free Clara?'),
            ('Who used it?', 'What saved Clara?', '', 'Who saved Clara?'),
            ('who used it?', 'what saved clara?', '', 'Who saved clara?'),
            ('Who used it?', 'How did Clara get to the island?', 'a compass', 'Who brought Clara to the island?'),
            ('Why was she hiding?', 'Who hid under the stairs?', '', 'Why was someone hiding under the stairs?'),
            ('Why was he studying?', 'Who studied in the hall?', '', 'Why was someone studying in the hall?'),
            ('Where was it actually?', 'What did Clara pretend to pull out of a hat?', 'a note',
             'Where did Clara really pull the item from?'),
            ('Where did Clara lose them?', 'Which sailor carried the blue flags?', 'the captain',
             'Where did the blue flags go when Clara lost them?'),
        ]
        for question, previous, answer, expected in cases:
            with self.subTest(question=question, previous=previous):
                self.assertEqual(followup_question(question, previous, answer), expected)

    def test_rewrites_pronouns_lowercase_names_and_short_followups(self):
        cases = [
            ('Who helped her?', 'Where did Dorothy go?', 'Who helped Dorothy?'),
            ('What was in her basket?', 'Where did Dorothy go?', "What was in Dorothy's basket?"),
            ('Where did he go?', 'What did jim pull?', 'Where did jim go?'),
            ('Where was it?', 'What did jim see?', 'Where was the item jim saw?'),
            ('Was it far?', 'What did the wizard go to see?', 'Was the item the wizard went to see far?'),
            ('What formed its tail?', 'What were the Sawhorse’s eyes made from?',
             "What formed the Sawhorse's tail?"),
            ('Why?', 'Who shod the Sawhorse with gold?', 'Why did someone shoe the Sawhorse with gold?'),
            ('What about Zeb?', "What color was Dorothy's face under the suns?",
             "What color was Zeb's face under the suns?"),
            ('What about the Wizard?', 'What did Jellia Jamb offer Jim to sleep in?',
             'What did Jellia Jamb offer the Wizard to sleep in?'),
            ('How did he fall into it?', 'What had the Braided Man invented for his business?',
             'How did the Braided Man fall into the thing the Braided Man invented?'),
        ]
        for question, previous, expected in cases:
            with self.subTest(question=question, previous=previous):
                session = BookSession(self.index, Mock(read=Mock(return_value=[])), trace_passages=True)
                session.remember(previous, '')
                self.assertEqual(session.reply(question).passage_trace.query, expected)

    def test_standalone_questions_with_her_and_that_start_fresh(self):
        questions = (
            'In her throne room, what did Ozma hold in her hand?',
            'Which gown did Dorothy wear to her party?',
            'What did Ozma see that made Dorothy laugh?',
        )
        for question in questions:
            with self.subTest(question=question):
                session = BookSession(self.index, Mock(read=Mock(return_value=[])), trace_passages=True)
                session.remember('Who is Dorothy?', '')
                reply = session.reply(question)
                self.assertEqual(reply.passage_trace.query, question)
                self.assertFalse(reply.history_used)


if __name__ == '__main__':
    unittest.main()
