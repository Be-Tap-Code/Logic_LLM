import re
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from fol_prover9_parser import Prover9_FOL_Formula
from formula import FOL_Formula

# set the path to the prover9 executable
# os.environ['PROVER9'] = '../Prover9/bin'
os.environ['PROVER9'] = '/home/manh/Logic-LLM/models/symbolic_solvers/Prover9/bin/prover9'

class FOL_Prover9_Program:
    def __init__(self, logic_program:str, dataset_name = 'FOLIO') -> None:
        self.logic_program = logic_program
        self.flag = self.parse_logic_program()
        self.dataset_name = dataset_name

    def parse_logic_program(self):
        try:        
            # Split the string into premises and conclusion
            premises_string = self.logic_program.split("Conclusion:")[0].split("Premises:")[1].strip()
            conclusion_string = self.logic_program.split("Conclusion:")[1].strip()

            # Extract each premise and the conclusion using regex
            premises = premises_string.strip().split('\n')
            conclusion = conclusion_string.strip().split('\n')

            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
            self.logic_premises = [p.replace('≡', '↔') for p in self.logic_premises]
            self.logic_premises = [p.replace('~', '¬') for p in self.logic_premises]
            self.logic_conclusion = conclusion[0].split(':::')[0].strip()

            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except:
            return False

    def execute_program(self):
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 10
            #prover = Prover9()
            #result = prover.prove(goal, assumptions)
            
            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()
            # print(prover.proof())
            if result:
                return 'True', ''
            else:
                # If Prover9 fails to prove, we differentiate between False and Unknown
                # by running Prover9 with the negation of the goal
                negated_goal = NegatedExpression(goal)
                # negation_result = prover.prove(negated_goal, assumptions)
                prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = prover.prove()
                if negation_result:
                    return 'False', ''
                else:
                    return 'Unknown', ''
        except Exception as e:
            return None, str(e)
        
    def answer_mapping(self, answer):
        if answer == 'True':
            return 'A'
        elif answer == 'False':
            return 'B'
        elif answer == 'Unknown':
            return 'C'
        else:
            raise Exception("Answer not recognized")
        
if __name__ == "__main__":
    # # ground-truth: True
    logic_program = """Predicates:\nEmployee(x) ::: x is an employee.\nSchedule(x) ::: x schedules a meeting with their customers.\nAppear(x) ::: x appears in the company.\nLunchCompany(x) ::: x has lunch in the company.\nLunchHome(x) ::: x has lunch at home.\nRemote(x) ::: x works remotely from home.\nOtherCountry(x) ::: x is in other countries.\nManager(x) ::: x is a manager.\nPremises:\n∀x (Employee(x) ∧ Schedule(x) → Appear(x)) ::: All employees who schedule a meeting with their customers will appear in the company today.\n∀x (LunchCompany(x) → Schedule(x)) ::: Everyone who has lunch in the company schedules meetings with their customers.\n∀x (Employee(x) → (LunchCompany(x) ⊕ LunchHome(x))) ::: Employees will either have lunch in the company or have lunch at home.\n∀x (LunchHome(x) → Remote(x)) ::: If an employee has lunch at home, then he/she is working remotely from home.\n∀x (Employee(x) ∧ OtherCountry(x) → Remote(x)) ::: All employees who are in other countries work remotely from home.\n∀x (Manager(x) → ¬Remote(x)) ::: No managers work remotely from home.\n(Manager(james) ∧ Appear(james)) ⊕ ¬(Manager(james) ∨ Appear(james)) ::: James is either a manager and appears in the company today or neither a manager nor appears in the company today.\nConclusion:\nLunchCompany(james) ::: James has lunch in the company."""

    # logic_program = "Predicates:\nOftenAttendEvents(x) ::: x often performs in school talent shows.\nEngagedWithSchoolEvents(x) ::: x is very engaged with school events.\nInactiveCommunityMember(x) ::: x is an inactive and disinterested member of his/her/its community.\nChaperoneDances(x) ::: x chaperones high school dances.\nStudentAttendingSchool(x) ::: x is a student attending the school.\nYoungChildOrTeenager(x) ::: x is a young child or teenager wishing to further his/her/its academic career and education.\nPremises:\nOftenAttendEvents(x) → EngagedWithSchoolEvents(x) ::: If people perform in school talent shows often, then they attend and are very engaged with school events.\nEither(OftenAttendEvents(x), InactiveCommunityMember(x)) ::: People either perform in school talent shows often or are inactive and disinterested members of their community.\nChaperoneDances(x) → ¬StudentAttendingSchool(x) ::: If people chaperone high school dances, then they are not students who attend the school.\nInactiveCommunityMember(x) → ChaperoneDances(x) ::: All people who are inactive and disinterested members of their community chaperone high school dances.\nYoungChildOrTeenager(x) → StudentAttendingSchool(x) ::: All young children and teenagers who wish to further their academic careers and educational opportunities are students who attend the school.\nBonnie(OftenAttendEvents(Bonnie), EngagedWithSchoolEvents(Bonnie), StudentAttendingSchool(Bonnie)) ⊕ ¬(OftenAttendEvents(Bonnie) ∨ EngagedWithSchoolEvents(Bonnie)) ⊕ ¬StudentAttendingSchool(Bonnie) ::: Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school.\n\nConclusion:\n\n(ChaperoneDances(Bonnie) ⊕ ¬ChaperoneDances(Bonnie)) → YoungChildOrTeenager(Bonnie) ∧ InactiveCommunityMember(Bonnie) ::: If Bonnie either chaperones high school dances or, if she does not, she performs in school talent shows often, then Bonnie is both a young child or teenager who wishes to further her academic career and educational opportunities and an inactive and disinterested member of the community.\n\n\n## Task Explanation\n\nIn this task you will be given some premises written in natural language followed by a question also written in natural language. You should interpret these inputs as sentences in First-Order Logic (FOL). Afterward, using standard FOL inference rules, determine whether the conclusion logically follows from the premise set. Your answer should be yes, no, or cannot say. \n\nYes means that it is possible to use the given premises to prove the conclusion using inference rules of FOL.\nNo means that there exists some interpretation where the premises are all true but the conclusion is false.\nCannot Say means that we cannot determine whether the conclusion can be proven from the premises because more context would need to be provided such as domain knowledge or additional facts.\n\n## Example\n\npremises: \n∀x (P(x) -> Q(x))\nQ(a)\n\nquestion: Is P(b)?\n\nanswer: Cannot Say"

    prover9_program = FOL_Prover9_Program(logic_program)

    answer, error_message = prover9_program.execute_program()
    print(answer)   