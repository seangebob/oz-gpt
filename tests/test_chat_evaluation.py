import unittest
from unittest.mock import Mock

import torch

from model.book_lookup import AnswerSpan, BookIndex, BookSession, NO_ANSWER, Passage
from model.chat_data import END_OF_TURN, ChatTokenizer, format_prompt, format_turn
from model.chat_evaluation import correctness, generate_predictions, reference_matches
from model.chat_session import ChatSession, DEFAULT_MAX_TOKENS, REPLY_RESERVE
from model.first_reply import FirstReplyClassifier, fit_first_replies
from model.model import block_size, chars


class ChatEvaluationTests(unittest.TestCase):
    def test_first_reply_classifier_selects_learned_answers_and_falls_back(self):
        classifier = FirstReplyClassifier(fit_first_replies([
            ('What does a sparrow eat?', 'Seeds.'), ('What is a sparrow fed?', 'Seeds.'),
            ('What does a panda eat?', 'Bamboo.'), ('What is a panda fed?', 'Bamboo.'),
        ]))
        self.assertEqual(classifier.predict('What do sparrows eat?'), 'Seeds.')
        self.assertEqual(classifier.predict('What do pandas eat?'), 'Bamboo.')
        self.assertIsNone(classifier.predict('12345'))

    def near_miss_classifiers(self):
        state = fit_first_replies([
            ('Who is Jim?', 'Jim is the cab-horse.'), ('What animal is Jim?', 'Jim is the cab-horse.'),
            ('Tell me about Jim.', 'Jim is the cab-horse.'), ('Who is Eureka?', "Eureka is Dorothy's kitten."),
            ("Who is Dorothy's kitten?", "Eureka is Dorothy's kitten."), ('What color was the lamp?', 'Blue.'),
            ('What color is the lantern?', 'Blue.'), ('Why did the lamp go out?', 'The oil ran out.'),
        ])
        legacy = {key: value for key, value in state.items() if key not in {'answer_terms', 'answer_question_words'}}
        return FirstReplyClassifier(state), FirstReplyClassifier({**legacy, 'first_reply_format': 1})

    def test_classifier_abstains_when_only_one_key_word_fits_the_answer(self):
        classifier, legacy = self.near_miss_classifiers()
        for question, answer in (('What size is Jim?', 'Jim is the cab-horse.'),
                                 ('Who is Dorothy?', "Eureka is Dorothy's kitten."),
                                 ('Where is Jim?', 'Jim is the cab-horse.'),
                                 ('Why is Jim?', 'Jim is the cab-horse.'),
                                 ("Where's Eureka?", "Eureka is Dorothy's kitten."),
                                 ('And where is Jim?', 'Jim is the cab-horse.'),
                                 ('Jim, where is he?', 'Jim is the cab-horse.'),
                                 ('And is it Jim?', 'Jim is the cab-horse.')):
            with self.subTest(question=question):
                self.assertEqual(legacy.predict(question), answer)
                self.assertIsNone(classifier.predict(question))

    def test_classifier_keeps_learned_questions_and_short_names(self):
        classifier, _ = self.near_miss_classifiers()
        self.assertEqual(classifier.predict('Who is Jim?'), 'Jim is the cab-horse.')
        self.assertEqual(classifier.predict('What animal is Jim?'), 'Jim is the cab-horse.')
        self.assertEqual(classifier.predict('Who is Eureka?'), "Eureka is Dorothy's kitten.")
        # A single name is answered when asked the way the answer was taught, or on its own.
        for question in ('Who was Jim?', "Who's Jim?", 'Jim?', 'jim', 'And who is Jim?', 'And Jim?'):
            with self.subTest(question=question):
                self.assertEqual(classifier.predict(question), 'Jim is the cab-horse.')
        self.assertIsNone(classifier.predict('Why?'))

    def test_classifier_rejects_invalid_answer_terms_and_question_words(self):
        state = fit_first_replies([('Who is Jim?', 'A horse.'), ('Who is Eureka?', 'A kitten.')])
        with self.assertRaisesRegex(ValueError, 'answer terms'):
            FirstReplyClassifier({**state, 'answer_terms': [['jim']]})
        self.assertEqual(state['answer_question_words'], [['who'], ['who']])
        for words in ([['who']], [['who'], ['jim']], [['who'], 'who']):
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, 'answer question words'):
                FirstReplyClassifier({**state, 'answer_question_words': words})

    def test_classifier_answers_only_the_first_turn_in_each_history_mode(self):
        tokenizer = ChatTokenizer(chars)
        conversations = [(1, {'messages': [
            {'role': 'user', 'content': 'Who is Jim?'},
            {'role': 'assistant', 'content': 'A horse.'},
            {'role': 'user', 'content': 'What kind?'},
            {'role': 'assistant', 'content': 'A cab-horse.'},
        ]})]
        for mode in ('generated', 'none', 'reference'):
            with self.subTest(mode=mode):
                model, classifier = Mock(), Mock()
                classifier.predict.return_value = 'A horse.'

                def generate(context, *_args, **_kwargs):
                    return torch.cat((context, torch.tensor([tokenizer.encode('A cab-horse.' + END_OF_TURN)])), dim=1)

                model.generate.side_effect = generate
                rows = generate_predictions(model, tokenizer, conversations, mode, DEFAULT_MAX_TOKENS,
                                            first_reply_classifier=classifier)
                classifier.predict.assert_called_once_with('Who is Jim?')
                self.assertEqual([row['reply_method'] for row in rows], ['classifier', 'generator'])
                self.assertEqual([row['clean_stop'] for row in rows], [None, True])
                self.assertEqual(rows[1]['history_used'], mode != 'none')
                self.assertTrue(all(row['reference_match'] for row in rows))

    def test_book_assistance_preserves_classifier_and_its_followup(self):
        tokenizer = ChatTokenizer(chars)
        book = Mock(spec=BookSession, previous_source=None)
        book.outside_names.return_value = []
        model, classifier = Mock(), Mock()
        classifier.predict.return_value = 'A horse.'

        def generate(context, *_args, **_kwargs):
            return torch.cat((context, torch.tensor([tokenizer.encode('A cab-horse.' + END_OF_TURN)])), dim=1)

        model.generate.side_effect = generate
        session = ChatSession(model, tokenizer, first_reply_classifier=classifier, book_session=book)

        first = session.reply('Who is Jim?')
        second = session.reply('What kind?')
        third = session.reply('What happened next?')

        self.assertEqual((first.reply_method, second.reply_method, third.reply_method),
                         ('classifier', 'generator', 'generator'))
        self.assertTrue(second.history_used)
        book.reply.assert_not_called()
        book.remember.assert_called_with('What happened next?', 'A cab-horse.')

    def test_cited_first_reply_can_override_an_unrelated_classifier_answer(self):
        tokenizer = ChatTokenizer(chars)
        passage = Passage('Chapter 1', 10, 14, 'Dorothy will carry a basket to the garden. Her basket was red.')
        book = BookSession(Mock(search=Mock(return_value=[(passage, 2.0)])),
                           Mock(read=Mock(return_value=[AnswerSpan('a basket', 7)])), trace_passages=True)
        classifier = Mock(predict=Mock(return_value='A lantern.'))
        session = ChatSession(Mock(), tokenizer, first_reply_classifier=classifier, book_session=book)

        reply = session.reply('What did Dorothy carry to the garden?', remember=False)
        self.assertEqual((reply.text, reply.reply_method, reply.source), ('a basket', 'book', passage))
        self.assertIsNone(book.previous_source)
        self.assertFalse(session.history)

        session.reset()
        reply = session.reply('What did Dorothy carry to the garden?')
        self.assertEqual(reply.passage_trace.query, 'What did Dorothy carry to the garden?')
        self.assertEqual(book.previous_question, 'What did Dorothy carry to the garden?')
        self.assertEqual(book.previous_source, passage)
        self.assertEqual(list(session.history), [('What did Dorothy carry to the garden?', 'a basket')])

    def test_weakly_related_or_equivalent_book_answers_keep_classifier_and_its_history(self):
        tokenizer = ChatTokenizer(chars)
        passages = [
            (Passage('Chapter 2', 30, 34, 'Dorothy met Zeb.'), 'Zeb'),
            (Passage('Chapter 1', 10, 14, 'Dorothy will carry a basket to the garden.'), 'basket'),
        ]
        classifier = Mock(predict=Mock(return_value='A basket.'))
        for passage, text in passages:
            with self.subTest(text=text):
                book = BookSession(Mock(search=Mock(return_value=[(passage, 2.0)])),
                                   Mock(read=Mock(return_value=[AnswerSpan(text, 7)])), trace_passages=True)
                session = ChatSession(Mock(), tokenizer, first_reply_classifier=classifier, book_session=book)

                reply = session.reply('What did Dorothy carry to the garden?')

                self.assertEqual((reply.text, reply.reply_method, reply.source), ('A basket.', 'classifier', None))
                self.assertIsNotNone(reply.passage_trace)
                self.assertIsNone(book.previous_source)
                self.assertEqual(book.previous_answer, 'A basket.')

    def test_confirmed_excerpt_can_override_classifier_when_the_wording_differs(self):
        index = BookIndex.from_text("CHAPTER 1.\n\nDorothy used a basket to carry the tools.")
        passage, = index.passages

        def read(question, passages):
            return [AnswerSpan('a basket', 9, start=p.text.index('a basket')) for p in passages]

        reader = Mock(read=Mock(side_effect=read))
        book = BookSession(index, reader)
        classifier = Mock(predict=Mock(return_value='A lantern.'))
        chat = ChatSession(Mock(), ChatTokenizer(chars), first_reply_classifier=classifier, book_session=book)

        result = chat.reply('What did Dorothy use to carry the tools up the hill?')

        self.assertEqual((result.text, result.reply_method, result.source), ('a basket', 'book', passage))
        self.assertIsNone(result.passage_trace)
        self.assertEqual(book.previous_source, passage)
        self.assertEqual(reader.read.call_count, 2)

    def test_reported_speech_routes_to_a_passage_with_the_requested_speaker(self):
        index = BookIndex.from_text(
            'CHAPTER 1.\n\n"The word means strength," said Zeb.\n\n'
            'CHAPTER 2.\n\n"The word means kindness," said Ozma.'
        )
        _, right = index.passages

        def read(question, passages):
            spans = []
            for passage in passages:
                wrong = passage.section == 'Chapter 1'
                text = 'strength' if wrong else 'kindness'
                spans.append(AnswerSpan(text, 13 if wrong else 9, start=passage.text.index(text)))
            return spans

        book = BookSession(index, Mock(read=Mock(side_effect=read)), trace_passages=True)
        chat = ChatSession(Mock(), ChatTokenizer(chars), book_session=book)

        result = chat.reply('What did Ozma say the word means?')

        self.assertEqual((result.text, result.source), ('kindness', right))
        self.assertEqual(result.passage_trace.reread.passage, right)
        self.assertEqual(book.previous_source, right)

    def scene_chat(self, passages, answer_passage, answer):
        def read(_question, candidates):
            return [AnswerSpan(answer, 9, start=passage.text.index(answer)) if passage == answer_passage
                    else AnswerSpan('', -1) for passage in candidates]

        book = BookSession(BookIndex(passages), Mock(read=Mock(side_effect=read)), trace_passages=True)
        return ChatSession(Mock(), ChatTokenizer(chars), book_session=book)

    def test_scene_name_can_appear_elsewhere_in_the_cited_chapter(self):
        opening = Passage('Chapter 2', 1, 2, 'Zeb entered the glass hall.')
        cited = Passage('Chapter 2', 5, 6, 'The Prince wore a star on his brow.')
        chat = self.scene_chat([opening, cited], cited, 'brow')

        result = chat.reply('Where did the Prince wear the star when he questioned Zeb?')

        self.assertEqual((result.text, result.reply_method, result.source), ('brow', 'book', cited))
        chat.model.generate.assert_not_called()

    def test_scene_name_missing_from_the_cited_chapter_still_abstains(self):
        cited = Passage('Chapter 2', 5, 6, 'The Prince wore a star on his brow.')
        elsewhere = Passage('Chapter 5', 9, 10, 'Zeb fed the horse.')
        chat = self.scene_chat([cited, elsewhere], cited, 'brow')

        result = chat.reply('Where did the Prince wear the star when he questioned Zeb?')

        self.assertEqual((result.text, result.reply_method, result.source), (NO_ANSWER, 'abstain', None))

    def test_common_noun_reporter_is_not_a_required_scene_name(self):
        cited = Passage('Chapter 6', 1, 2, 'From above, the vines looked like snakes.')
        chat = self.scene_chat([cited], cited, 'snakes')

        result = chat.reply('What did the group say the vines looked like?')

        self.assertEqual((result.text, result.reply_method, result.source), ('snakes', 'book', cited))

    def memory_chat(self, model=None):
        passages = [
            Passage('Chapter 1', 1, 2, 'The Wizard was born in Omaha.'),
            Passage('Chapter 2', 3, 4, 'Zeb stood beside Ozma.'),
            Passage('Chapter 3', 5, 6, 'Zeb ran to the barn.'),
        ]
        answers = {
            'Where was the Wizard born?': 'Omaha',
            'Where did Zeb hide?': 'Ozma',
            'What did Zeb do next?': 'ran',
        }

        def read(question, candidates):
            answer = answers.get(question, '\0')
            return [AnswerSpan(answer, 9, start=passage.text.index(answer)) if answer in passage.text
                    else AnswerSpan('', -1) for passage in candidates]

        book = BookSession(BookIndex(passages), Mock(read=Mock(side_effect=read)), trace_passages=True)
        return ChatSession(model or Mock(), ChatTokenizer(chars), book_session=book), book, passages

    def test_followup_after_abstention_uses_the_abstained_subject(self):
        chat, book, passages = self.memory_chat()
        self.assertEqual(chat.reply('Where was the Wizard born?').text, 'Omaha')

        abstained = chat.reply('Where did Zeb hide?')

        self.assertEqual((abstained.text, abstained.reply_method), (NO_ANSWER, 'abstain'))
        self.assertEqual((book.previous_question, book.previous_answer, book.previous_source),
                         ('Where did Zeb hide?', '', None))
        self.assertFalse(chat.history)

        followup = chat.reply('What did he do next?')

        self.assertEqual(followup.passage_trace.query, 'What did Zeb do next?')
        self.assertEqual((followup.text, followup.source), ('ran', passages[2]))
        chat.model.generate.assert_not_called()

    def test_abstention_is_not_remembered_when_history_is_disabled(self):
        chat, book, passages = self.memory_chat()
        chat.reply('Where was the Wizard born?')

        self.assertEqual(chat.reply('Where did Zeb hide?', remember=False).reply_method, 'abstain')

        self.assertEqual((book.previous_question, book.previous_source), ('Where was the Wizard born?', passages[0]))
        self.assertEqual(len(chat.history), 1)

    def test_unfinished_generator_reply_updates_the_book_subject(self):
        tokenizer = ChatTokenizer(chars)
        model = Mock()
        model.generate.side_effect = lambda context, _limit, **_options: torch.cat(
            (context, torch.tensor([tokenizer.encode('Unfinished')])), dim=1,
        )
        chat, book, _ = self.memory_chat(model)
        chat.reply('Where was the Wizard born?')

        reply = chat.reply('Who is Ozma?')

        self.assertFalse(reply.ended_with_marker)
        self.assertEqual((book.previous_question, book.previous_answer, book.previous_source),
                         ('Who is Ozma?', '', None))
        self.assertFalse(chat.history)
        chat.reply('What did she do?')
        self.assertEqual(book.previous_question, 'What did Ozma do?')

    def test_question_about_a_name_outside_the_book_abstains_without_changing_memory(self):
        chat, book, passages = self.memory_chat()
        self.assertEqual(chat.reply('Where was the Wizard born?').text, 'Omaha')

        for question in ('Who wrote Hamlet?', 'How tall is Mount Everest?', 'Tell me about Shakespeare.'):
            with self.subTest(question=question):
                reply = chat.reply(question)

                self.assertEqual((reply.text, reply.reply_method, reply.source), (NO_ANSWER, 'abstain', None))
                chat.model.generate.assert_not_called()
                self.assertEqual((book.previous_question, book.previous_source),
                                 ('Where was the Wizard born?', passages[0]))
                self.assertEqual(len(chat.history), 1)

    def test_trained_first_reply_still_answers_a_question_with_an_outside_name(self):
        chat, _, _ = self.memory_chat()
        chat.first_reply_classifier = Mock(predict=Mock(return_value='I can help with questions about Oz.'))

        reply = chat.reply('What is the capital of France?')

        self.assertEqual((reply.text, reply.reply_method), ('I can help with questions about Oz.', 'classifier'))

    def generator(self, *replies):
        tokenizer = ChatTokenizer(chars)
        replies = iter(replies)
        return Mock(generate=Mock(side_effect=lambda context, _limit, **_options: torch.cat(
            (context, torch.tensor([tokenizer.encode(next(replies))])), dim=1,
        )))

    def test_unfinished_reply_without_a_named_question_keeps_the_book_subject(self):
        passages = [
            Passage('Chapter 3', 796, 824, 'The Sorcerer called it a Rain of People. Some stones fell with them.'),
            Passage('Chapter 5', 1365, 1387, 'The Prince said that his people fell from the sky.'),
        ]
        answers = {
            'What did the Sorcerer call the second rain?': 'a Rain of People',
            'What else did the Sorcerer say fell along with them?': 'Some stones',
            'What else did he say fell along with them?': 'his people',
        }

        def read(question, candidates):
            answer = answers.get(question, '\0')
            return [AnswerSpan(answer, 9, start=passage.text.index(answer)) if answer in passage.text
                    else AnswerSpan('', -1) for passage in candidates]

        middle_turns = ('Thanks.', 'Thanks, that makes sense.', 'I had forgotten that part!', 'Really?',
                        'Is that so?', 'How strange!', 'Why would anyone do that?')
        for middle_turn in middle_turns:
            with self.subTest(middle_turn=middle_turn):
                book = BookSession(BookIndex(passages), Mock(read=Mock(side_effect=read)))
                chat = ChatSession(self.generator('In Ozma'), ChatTokenizer(chars), book_session=book)
                self.assertEqual(chat.reply('What did the Sorcerer call the second rain?').text, 'a Rain of People')

                self.assertFalse(chat.reply(middle_turn).ended_with_marker)

                self.assertEqual((book.previous_question, book.previous_source),
                                 ('What did the Sorcerer call the second rain?', passages[0]))
                followup = chat.reply('What else did he say fell along with them?')
                self.assertEqual((followup.text, followup.source), ('Some stones', passages[0]))

    def test_unfinished_reply_without_a_named_question_keeps_an_unanswered_subject(self):
        chat, book, passages = self.memory_chat(self.generator('Unfinished'))
        chat.reply('Where was the Wizard born?')
        self.assertEqual(chat.reply('Where did Zeb hide?').reply_method, 'abstain')

        chat.reply('Thanks')

        self.assertEqual(book.previous_question, 'Where did Zeb hide?')
        followup = chat.reply('What did he do next?')
        self.assertEqual((followup.text, followup.source), ('ran', passages[2]))

    def test_unfinished_reply_keeps_history_without_book_assistance(self):
        tokenizer = ChatTokenizer(chars)
        replies = iter(('A horse.' + END_OF_TURN, 'Unfinished'))
        model = Mock()
        model.generate.side_effect = lambda context, _limit, **_options: torch.cat(
            (context, torch.tensor([tokenizer.encode(next(replies))])), dim=1,
        )
        chat = ChatSession(model, tokenizer)
        chat.reply('Who is Jim?')

        chat.reply('What kind?')

        self.assertEqual(list(chat.history), [('Who is Jim?', 'A horse.')])

    def test_bare_count_answer_to_a_size_question_abstains_without_remembering_it(self):
        index = BookIndex.from_text('CHAPTER 1.\n\nThree automatons stood in the hall.')
        reader = Mock(read=Mock(return_value=[AnswerSpan('Three', 10)]))
        book = BookSession(index, reader, trace_passages=True)
        model = Mock()
        chat = ChatSession(model, ChatTokenizer(chars), book_session=book)

        result = chat.reply('How big was the automaton in the hall?')

        self.assertEqual((result.text, result.reply_method, result.source), (NO_ANSWER, 'abstain', None))
        self.assertIsNone(book.previous_source)
        self.assertFalse(chat.history)
        model.generate.assert_not_called()

    def test_book_assistance_uses_cited_history_and_keeps_generator_fallback(self):
        tokenizer = ChatTokenizer(chars)
        passage = Passage('Chapter 1', 1, 4, 'Jim pulled a buggy. The buggy was red.')
        index = Mock(search=Mock(return_value=[(passage, 2.0)]))

        def read(question, passages):
            text = {'What did Jim pull?': 'a buggy', 'What color was the item Jim pulled?': 'red'}.get(question, '')
            return [AnswerSpan(text, 8) for _ in passages]

        book = BookSession(index, Mock(read=Mock(side_effect=read)), trace_passages=True)
        conversations = [(1, {'messages': [
            {'role': 'user', 'content': 'What did Jim pull?'},
            {'role': 'assistant', 'content': 'A buggy.'},
            {'role': 'user', 'content': 'What color was it?'},
            {'role': 'assistant', 'content': 'Red.'},
            {'role': 'user', 'content': 'Where is Zeb?'},
            {'role': 'assistant', 'content': 'At the ranch.'},
        ]})]
        model = Mock()

        def generate(context, *_args, **_kwargs):
            prompt = tokenizer.decode(context[0].tolist())
            self.assertIn('red', prompt)
            return torch.cat((context, torch.tensor([tokenizer.encode('At the ranch.' + END_OF_TURN)])), dim=1)

        model.generate.side_effect = generate
        rows = generate_predictions(model, tokenizer, conversations, 'generated', DEFAULT_MAX_TOKENS,
                                    book_session=book)

        self.assertEqual([row['reply_method'] for row in rows], ['book', 'book', 'generator'])
        self.assertEqual([row['clean_stop'] for row in rows], [None, None, True])
        self.assertTrue(rows[1]['history_used'])
        self.assertEqual(rows[1]['source']['section'], 'Chapter 1')
        self.assertEqual(rows[1]['passage_trace']['previous_answer'], 'a buggy')
        self.assertIsNone(rows[2]['source'])
        self.assertIsNone(rows[2]['passage_trace']['selected_rank'])
        self.assertEqual(book.previous_answer, 'At the ranch.')

    def test_history_modes_use_the_right_reply_and_reset_between_conversations(self):
        tokenizer = ChatTokenizer(chars)
        conversations = [
            (1, {'messages': [
                {'role': 'user', 'content': 'Who is Jim?'},
                {'role': 'assistant', 'content': 'A horse.'},
                {'role': 'user', 'content': 'What is his name?'},
                {'role': 'assistant', 'content': 'Jim.'},
            ]}),
            (3, {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi!'},
            ]}),
        ]
        histories = {
            'generated': format_turn('Who is Jim?', 'A kitten.'),
            'none': '',
            'reference': format_turn('Who is Jim?', 'A horse.'),
        }
        for mode, history in histories.items():
            with self.subTest(history=mode):
                contexts = []
                responses = iter(('A kitten.' + END_OF_TURN, 'JIM!' + END_OF_TURN, 'He'))

                def generate(context, _limit, **options):
                    self.assertFalse(options['sample'])
                    contexts.append(tokenizer.decode(context[0].tolist()))
                    return torch.cat((context, torch.tensor([tokenizer.encode(next(responses))])), dim=1)

                model = Mock()
                model.generate.side_effect = generate
                rows = generate_predictions(model, tokenizer, conversations, mode, DEFAULT_MAX_TOKENS)
                self.assertEqual(contexts, [
                    format_prompt('Who is Jim?'),
                    history + format_prompt('What is his name?'),
                    format_prompt('Hello'),
                ])
                self.assertEqual([(row['conversation'], row['turn']) for row in rows], [(1, 1), (1, 2), (3, 1)])
                self.assertEqual(rows[1]['history_used'], bool(history))
                self.assertFalse(any(row['history_compacted'] for row in rows))
                self.assertEqual([row['correct'] for row in rows], [None, True, None])
                self.assertEqual([row['clean_stop'] for row in rows], [True, True, False])

    def test_long_reply_is_shortened_when_including_the_latest_turn(self):
        tokenizer = ChatTokenizer(chars)
        first_prompt = 'Who is Jim?'
        first_reply = 'Jim is a horse. He pulls the buggy and travels with Dorothy and Zeb.'
        followup = 'What kind of animal?'
        contexts = []
        responses = iter((first_reply, 'A horse.'))

        def generate(context, max_tokens, **_options):
            contexts.append((tokenizer.decode(context[0].tolist()), max_tokens))
            return torch.cat((context, torch.tensor([tokenizer.encode(next(responses) + END_OF_TURN)])), dim=1)

        model = Mock()
        model.generate.side_effect = generate
        session = ChatSession(model, tokenizer)
        self.assertEqual(session.reply(first_prompt).text, first_reply)
        reply = session.reply(followup)

        self.assertTrue(reply.history_used)
        self.assertTrue(reply.history_compacted)
        self.assertEqual(reply.text, 'A horse.')
        prompt, remaining = contexts[1]
        self.assertEqual(prompt, format_turn(first_prompt, 'Jim is a horse. He') + format_prompt(followup))
        self.assertGreaterEqual(remaining, REPLY_RESERVE)
        self.assertEqual(len(prompt) + remaining, block_size + 1)

    def test_reviewed_correctness_is_distinct_from_reference_matches(self):
        self.assertTrue(reference_matches('  THE   Magic Belt! ', 'The Magic Belt.'))
        self.assertFalse(reference_matches('The enchanted belt.', 'The Magic Belt.'))
        self.assertEqual(correctness([{'correct': None}]), 'pending (1 to review)')
        self.assertEqual(correctness([{'correct': True}, {'correct': None}]), '1/1 reviewed (1 pending)')
        self.assertEqual(correctness([{'correct': True}, {'correct': False}]), '1/2')
        with self.assertRaisesRegex(ValueError, 'true, false, or null'):
            correctness([{'correct': 1}])


if __name__ == '__main__':
    unittest.main()
