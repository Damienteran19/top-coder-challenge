# Interview Notes

## Marcus
* More (reciepts/day) -> less reimbursement
* Reimbursement to mileage graph is non-linear (tapers off)
* More (miles/day) -> more reimbursement

## Lisa
* Reimbursement to mileage graph has a slope of 0.58 $/mile for first 100 miles
* Reimbursement to days graph is mostly linear, with slope of 100 $/day, most times
* Reciepts get reimbursed, up until a soft cap (around $600-$800 of receipts) where reimbursement to receipt graph has terribly small slope
    * However, low receipt amount ($50-$80 of receipts) -> less reimbursement
* more (miles/day) -> more reimbursement
    * However, there is no linear relationship between miles/day and reimbursement, other factors at play
* receipts end in $0.49 or $0.99 -> more reimbursement (rounding errors?)
* Two instances with "same type of trip" get different reimbursement?

## Dave
* 4 days, 100 miles, X receipts -> really good reimbursement or way worse reimbursement
* Less mileage -> more reimbursement/mile OR more reimbursement
* 7 days, lots miles, decent receipts -> ok reimbursement
* submit no receipts -> base reimbursement/day
    * submit a small total receipts(~$12) -> less reimbursement/day
* more receipts -> more % reimbursed of the receipts
* Magic Combination of 3 raw features produces the most reimbursement?

## Jennifer
* There are hidden factors that make system seem nepotic, and can catch new employeesin "pitfalls"
* Older employees generally get better reimbursement
* Submit a small total receipts -> less reimbursement
* Days from 4-6 -> more reimbursement
* Lots of miles -> more reimbursement
* Conservative spending -> more reimbursement

## Kevin
* Efficiency is huge
* 180-220 miles/day -> more reimbursement (maximum of reimbursement-(miles/day) graph?)
* =<3 days and <$75 receipt/day -> more reimbursement
* 4-6 days and <$120 receipt/day -> more reimbursement
* =>7 days and <$90 receipt/day -> more reimbursement
* System may be branched, different equations for at least 6 kinds of instances
    * low days and high mileage, high days and low mileage, or "medium" days and "balanced" mileage are potential branches (for a 15% accurate model)
* miles * (receipt/day), "Trip length times efficiency", and other "iteractive" derived features may play a large role
* Potential thresholds for bonuses/penalties to reimbursement total?
    * =>8 days and high spending -> reimbursement penalty
    * 5-days and 180+ miles/day and <$100 receipts/day -> reimbursement bonus
* high mileage and low spending -> more reimbursement
* low mileage and high spending -> less reimbursement
