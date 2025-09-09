import argparse
from components.utils import FileIO

from sklearn.metrics import accuracy_score, f1_score


class Evaluator:


    def __init__(
        self, 
        eval_method: str, 
        timetom_eval_method: str, 
        eval_prompt: dict, 
        data_name: str, 
        add_label: bool = False,
    ):
        self.eval_method = eval_method
        self.eval_prompt = eval_prompt
        self.data_name = data_name
        self.add_label = add_label
        self.timetom_eval_method = timetom_eval_method

    
    @staticmethod
    def compute_lexical_overlap(pred: str, location: str) -> float:
        pred = pred.lower().replace('_', ' ') \
            .replace("'s", '') \
            .replace('.', '') \
            .replace('"', '') \
            .replace("'", '') \
            .replace('[', '') \
            .replace(']', '')
        location = location.lower().replace('_', ' ') \
            .replace("'s", '') \
            .replace('.', '') \
            .replace('"', '') \
            .replace("'", '') \
            .replace('[', '') \
            .replace(']', '')

        score = 0 
        pred_tokens = pred.replace('.', '').split()
        location_tokens = location.split()
        visited_tokens = []

        for token in pred_tokens:
            if token in location_tokens and token not in visited_tokens:
                score += 1
                visited_tokens.append(token)

        return score / len(location)


    @staticmethod
    def __get_candidate_answers_for_opentom(cur_question: str, candidate_answers: dict) -> list:
        if 'initial location' in cur_question:
            candidate_key = 'location_coarse'
        elif 'precisely' in cur_question:
            candidate_key = 'location_fine'
        elif 'attitude' in cur_question:
            candidate_key = 'attitude'
        elif 'fullness' in cur_question:
            candidate_key = 'fullness'
        elif 'accessibility' in cur_question:
            candidate_key = 'accessibility'
        else:
            candidate_key = 'location_fine'
        
        return candidate_answers[candidate_key]


    @staticmethod
    def normalize_token(word: str) -> str:
        determinants = ['a', 'an', 'the']
        for det in determinants:
            if word.startswith(det):
                word = word[len(det):].strip()

        word = word.lower() \
            .replace("'s", '') \
            .replace('.', '') \
            .replace('"', '') \
            .replace("'", '') \
            .strip()
        return word


    def evaluate_free_form(
        self,
        questions: list,
        pred: list,
        gt:list,
        container_lst: list = [],
    ) -> tuple:
        """
        evaluate the answer from free-form response 
        Datasets: ToMi
        """

        # return "correct" if the predicted answer is correct, otherwise return "incorrect"
        for q_dict_idx, q_dict in enumerate(questions):

            try:
                pred_token = q_dict['predicted']
            except:
                continue
            # split for CoT
            pred_token = pred_token.split('Therefore')[-1]
            # split for Vanilla and CoT
            pred_token = pred_token.split('<answer>')[-1]
            pred_token = pred_token.split('</answer>')[0]

            gt_token = q_dict['answer'] 

            pred_token, gt_token = str(pred_token).strip(), str(gt_token).strip()
            pred_token = self.normalize_token(pred_token.lower())
            gt_token = self.normalize_token(gt_token.lower())

            # if self.data_name == 'tomi':
            candidate_answers = [ele for ele in container_lst if ele != gt_token]
            # elif self.data_name == 'opentom':
            #     candidate_answers = self.eval_prompt['opentom']['candidate_answers']
            #     candidate_answers = self.__get_candidate_answers_for_opentom(
            #         q_dict['question'], 
            #         candidate_answers
            #     )
            #     candidate_answers = [ele for ele in candidate_answers if ele != gt_token]

            if candidate_answers:
                gt_score = self.compute_lexical_overlap(pred_token, gt_token)
                other_scores = []
                for candidate in candidate_answers:
                    other_scores.append(
                        self.compute_lexical_overlap(pred_token, str(candidate).strip())
                    )

                if gt_score > max(other_scores):
                    cur_pred = 1
                else:
                    cur_pred = 0

                gt.append(1)

            else:
                if gt_token in pred_token:
                    cur_pred = 1
                else:
                    cur_pred = 0

                gt.append(1)

            pred.append(cur_pred)

            if self.add_label:
                questions[q_dict_idx]['label'] = cur_pred

        return gt, pred, questions


    def evaluate_multiple_choice(
        self, 
        questions: list,
        pred: list,
        gt: list,
        data_name: str,
    ) -> tuple:
        """
        evaluate the answer from multiple choice response
        Datasets: BigToM, HiToM 
        """

        for q_dict_idx, q_dict in enumerate(questions):

            try:
                predicted = q_dict['predicted']
            except:
                continue

            # # if LLM generation followed instruction, parse the answer inbetween the tags
            # if '<answer>' in predicted:
            #     predicted = predicted.split('</answer>')[0].split('<answer>')[-1].strip().lower()
            #     predicted_lst = [ele.split('<answer>')[-1].strip().lower() for ele in predicted]
            #     predicted_lst = [ele for ele in predicted_lst if ele.strip()]
            # # if LLM generation is for properly formatted, split by colon or period character
            # else:
            #     predicted_lst = predicted.rsplit(':')[-1].strip().split('.', 1)[-1].lower()
            #     # continue
            #
            # # retrieve correct answer
            if 'correct_letter' in q_dict.keys():
                answer = q_dict['correct_letter']
                answer = answer.lower() if isinstance(answer, str) else answer
            elif 'answer' in q_dict.keys():
                answer = q_dict['answer'].lower()
            elif 'true_answer' in q_dict.keys():
                answer = q_dict['true_answer'].lower()
            else:
                raise ValueError('No answer key found in the question dictionary')

            if '_' in answer:
                answer = [answer.lower(), answer.replace('_', ' ').lower()]

            # Handle questions of different formats in FANToM
            if data_name == 'fantom':

                if '<answer>' in predicted:
                    predicted = predicted.split('</answer>')
                    predicted = [ele for ele in predicted if ele.strip()]
                    if len(predicted) > 1:
                        predicted = [ele.split('<answer>')[-1].strip().lower() for ele in predicted]
                        predicted = [ele for ele in predicted if ele.strip()]
                    else:
                        predicted = predicted[0].split('<answer>')[-1].strip().lower()
                        predicted = predicted.split('.', 1)[0].lower()
                # if LLM generation is for properly formatted, split by colon or period character
                else:
                    predicted = predicted.rsplit(':')[-1].strip().split('.', 1)[-1].lower()


                correct_answer = q_dict['correct_answer']
                # InfoAccess in FANToM
                if isinstance(correct_answer, list):
                    answer = [ele.lower() for ele in answer]
                    if isinstance(predicted, str):
                        predicted = predicted.split(',')

                    predicted = [ele.strip().replace('"', '').replace("'", '') for ele in predicted]
                    predicted = [ele.lower() for ele in predicted if ele.strip()]

                    answer = sorted(answer)
                    predicted = sorted(predicted)

                    if answer == predicted:
                        cur_pred = 1
                    else:
                        cur_pred = 0

                # Binary questions in FANToM
                else:
                    if answer == predicted:
                        cur_pred = 1
                    else:
                        cur_pred = 0
            else:
                if '<answer>' in predicted:
                    predicted = predicted.split('</answer>')[0].split('<answer>')[-1].strip().lower()
                else:
                    predicted = predicted.rsplit(':')[0].strip().split('.', 1)[0].lower()

                predicted = predicted.split('.')[0].strip().lower()

                if predicted == answer:
                # if any([a in predicted for a in answer]):
                    cur_pred = 1
                else:
                    cur_pred = 0

            pred.append(cur_pred)
            gt.append(1)

            if self.add_label:
                questions[q_dict_idx]['label'] = cur_pred

        return gt, pred, questions


    def evaluate(
        self, 
        result_data: dict, 
        data_name: str,
        timetom: bool = False, 
        mc_probing: bool = False,
    ) -> tuple:

        if mc_probing:
            eval_method = 'multiple-choice'
        elif timetom:
            eval_method = self.timetom_eval_method
        else:
            eval_method = self.eval_method

        gt, pred = [], []
        
        all_questions = {}
        for key, val in result_data.items():
            questions = val['questions']

            container_lst = []
            if 'containers' in val.keys():
                container_lst = val['containers']

            if eval_method == 'free-form':
                gt, pred, questions = self.evaluate_free_form(
                    questions=questions, 
                    pred=pred, 
                    gt=gt, 
                    container_lst=container_lst
                )

            elif eval_method == 'multiple-choice':
                gt, pred, questions = self.evaluate_multiple_choice(
                    questions=questions, 
                    pred=pred, 
                    gt=gt,
                    data_name=data_name,
                )

            # # update question by adding the label for the prediction 
            # # 0 <- incorrect, 1 <- correct
            # if self.add_label:
            #     result_data[key]['questions'] = questions

            all_questions[key] = questions
            
        assert len(gt) == len(pred)

        return gt, pred, all_questions
