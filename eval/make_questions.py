"""Rebuilds eval/enterprise_questions.jsonl (adds the dev/test split). Run once: python eval/make_questions.py"""
import json, os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enterprise_questions.jsonl")
old = [json.loads(l) for l in open(P)]
old = [q for q in old if "split" not in q] or old
def A(i, q, g, role="employee"): return dict(id=f"ans{i}", type="answerable", role=role, question=q, gold=[g])
new = [
 A(29,"Within how many days must a grievance be raised in writing?","60 days"),
 A(30,"Within how many working days does HR respond to a grievance?","10 working days"),
 A(31,"Within how many days can an employee appeal a grievance outcome?","14 days"),
 A(32,"What value of gift or hospitality must be declared to a manager?","50 dollars"),
 A(33,"Within how many working days must a conflict of interest be declared?","5 working days"),
 A(34,"How quickly is a whistleblowing report acknowledged?","24 hours"),
 A(35,"In which months do performance reviews take place?","June and December"),
 A(36,"What is the rating scale used in performance reviews?","1 to 5"),
 A(37,"How long does a performance improvement plan last?","90 days"),
 A(38,"How many days before the review meeting is the self-assessment form due?","7 days"),
 A(39,"Above what amount do purchases need three written quotes?","2,000 dollars"),
 A(40,"Above what amount does a purchase order need director sign-off?","10,000 dollars"),
 A(41,"What is the standard payment term for suppliers?","30 days"),
 A(42,"At what time does the office open?","07:30"),
 A(43,"How far ahead can meeting rooms be booked?","14 days"),
 A(44,"After how many minutes are unclaimed meeting room bookings released?","15 minutes"),
 A(45,"How long are system logs retained?","90 days"),
 A(46,"How long are backups kept?","35 days"),
 A(47,"How soon is customer data deleted after a contract ends?","30 days"),
 A(48,"After how long is email in shared mailboxes archived?","2 years"),
 A(49,"What does a promotion require in terms of ratings?","two consecutive ratings of 4 or higher","hr"),
 A(50,"Above what foreign exchange exposure is hedging required?","500,000 dollars","finance"),
 A(51,"Investments above what amount need Chief Financial Officer approval?","250,000 dollars","finance"),
 A(52,"How often does the promotion committee meet?","quarterly","hr"),
]
def U(i, q): return dict(id=f"una{i}", type="unanswerable", role="employee", question=q, gold=[])
new += [U(15,"What is the notice period for resignation?"),U(16,"What is the overtime pay rate?"),U(17,"How much is the employee referral bonus?"),
 U(18,"What is the pension contribution percentage?"),U(19,"How many days of study leave are provided?"),U(20,"How long are CCTV recordings kept?"),
 U(21,"What is the penalty for paying a supplier late?"),U(22,"What is the price of lunch in the cafeteria?"),
 U(23,"How many guests can one employee bring to the office per day?"),U(24,"What is the monthly health insurance premium?")]
def R(i, q, g, role="employee"): return dict(id=f"res{i}", type="restricted", role=role, question=q, gold=[g])
new += [R(9,"What share of employees can receive the top performance rating?","10 percent"),R(10,"In which month are calibration meetings held?","January"),
 R(11,"What is the daily cash movement limit?","2,000,000 dollars"),R(12,"Above what foreign exchange exposure is hedging required?","500,000 dollars"),
 R(13,"Who must co-sign investments above 250,000 dollars?","second signatory","hr")]
allq = old + new
for t in ("answerable", "unanswerable", "restricted"):
    grp = [q for q in allq if q["type"] == t]
    for k, q in enumerate(grp): q["split"] = "dev" if k % 5 in (0, 2) else "test"
with open(P, "w") as fh: fh.write("\n".join(json.dumps(q) for q in allq) + "\n")
from collections import Counter; print(len(allq), Counter((q["type"], q["split"]) for q in allq))
