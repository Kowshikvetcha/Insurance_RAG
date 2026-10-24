/** Common questions people ask before buying a health insurance policy, grouped for the FAQ panel. */

export interface FaqCategory {
  id: string
  label: string
  hint: string
  questions: string[]
}

export const FAQ: FaqCategory[] = [
  {
    id: 'coverage',
    label: 'Coverage',
    hint: 'What the policy pays for',
    questions: [
      'What does this policy cover? Give me a summary of the main benefits.',
      'Is maternity covered, and with what limits and waiting period?',
      'Is a newborn baby covered from birth?',
      'Are pre- and post-hospitalisation expenses covered, and for how many days?',
      'Are day-care procedures covered?',
      'Is AYUSH (Ayurveda, homeopathy etc.) treatment covered?',
      'Are OPD consultations and medicines covered?',
      'Is road or air ambulance covered, and up to how much?',
      'Are mental illness treatments covered?',
      'Are modern treatments like robotic surgery covered?',
    ],
  },
  {
    id: 'waiting',
    label: 'Waiting periods',
    hint: 'When cover starts',
    questions: [
      'What is the initial waiting period after buying the policy?',
      'What is the waiting period for pre-existing diseases?',
      'Which specific diseases have a waiting period, and how long is it?',
      'Are accidents covered from day one?',
      'Can any waiting period be reduced or waived?',
    ],
  },
  {
    id: 'limits',
    label: 'Limits & costs',
    hint: 'Caps, co-pay and bonuses',
    questions: [
      'Is there a room rent limit? Give it for each plan.',
      'Is there an ICU charge limit?',
      'Is there any co-payment, and when does it apply?',
      'Is there a deductible?',
      'Which treatments have sub-limits (e.g. cataract, knee replacement)?',
      'What happens if I choose a room above my eligible limit?',
      'Is the sum insured restored if it gets used up?',
      'Is there a no-claim or cumulative bonus, and how much?',
    ],
  },
  {
    id: 'exclusions',
    label: 'Exclusions',
    hint: 'What is not covered',
    questions: [
      'What are the permanent exclusions?',
      'Is cosmetic or plastic surgery covered?',
      'Are injuries from adventure sports covered?',
      'Is obesity or weight-loss treatment covered?',
      'Are dental treatments covered?',
      'Are self-inflicted injuries or substance abuse excluded?',
    ],
  },
  {
    id: 'claims',
    label: 'Claims',
    hint: 'How to get paid',
    questions: [
      'How do I make a cashless claim?',
      'How do I file a reimbursement claim, and what documents are needed?',
      'How soon must I inform the insurer or TPA before or after hospitalisation?',
      'What is the deadline to submit claim documents after discharge?',
      'Can I get treatment at a non-network hospital?',
      'What is a proportionate deduction and when does it apply?',
    ],
  },
  {
    id: 'renewal',
    label: 'Renewal & premium',
    hint: 'Price, renewal, cancellation',
    questions: [
      'How is the premium calculated, and how does it change with age?',
      'Is the policy renewable for life?',
      'Is there a grace period for paying the renewal premium?',
      'What is the free-look period, and can I cancel for a refund?',
      'Can I port this policy to another insurer?',
      'Are there any premium discounts available?',
    ],
  },
]
