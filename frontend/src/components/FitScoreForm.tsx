import { useState } from 'react'
import type { HealthProfile, Member } from '../api'

export const EMPTY_PROFILE: HealthProfile = {
  age: 30,
  gender: 'unspecified',
  city_tier: 'unspecified',
  covered_members: ['self'],
  desired_sum_insured: null,
  annual_budget: null,
  conditions: [],
  medications: '',
  past_surgeries: '',
  hospitalised_last_2y: false,
  smoker: false,
  alcohol: 'none',
  height_cm: null,
  weight_kg: null,
  hazardous_activity: '',
  planning_pregnancy: false,
  planned_procedures: [],
  wants_opd: false,
  wants_dental: false,
  wants_ayush: false,
  avoid_copay: false,
  avoid_room_rent_limit: false,
}

const COMMON_CONDITIONS = [
  'Diabetes',
  'Hypertension',
  'Heart disease',
  'Asthma / COPD',
  'Thyroid disorder',
  'Cancer (history)',
  'Kidney disease',
  'Liver disease',
  'Arthritis / joint problems',
  'Mental health condition',
]
const MEMBERS: { id: Member; label: string }[] = [
  { id: 'self', label: 'Myself' },
  { id: 'spouse', label: 'Spouse' },
  { id: 'children', label: 'Children' },
  { id: 'parents', label: 'Parents' },
]

const splitList = (s: string) =>
  s
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)

const numOrNull = (s: string) => (s.trim() === '' ? null : Number(s))

/** "₹5 lakh" / "₹1.5 crore" hint under the rupee inputs, so a misplaced zero is easy to spot. */
const inWords = (n: number | null) => {
  if (!n || !Number.isFinite(n)) return ''
  if (n >= 1e7) return `= ₹${+(n / 1e7).toFixed(2)} crore`
  if (n >= 1e5) return `= ₹${+(n / 1e5).toFixed(2)} lakh`
  return `= ₹${n.toLocaleString('en-IN')}`
}

interface Props {
  initial: HealthProfile
  busy: boolean
  serverError?: string | null // the server rejected the last submission
  onSubmit: (p: HealthProfile) => void
}

/** The health profile form. Values live in React state only; nothing is saved in the browser. */
export function FitScoreForm({ initial, busy, serverError, onSubmit }: Props) {
  const [p, setP] = useState<HealthProfile>(initial)
  // Free-text extras are kept as typed and merged into the lists on submit.
  const [otherConditions, setOtherConditions] = useState(
    initial.conditions.filter((c) => !COMMON_CONDITIONS.includes(c)).join(', '),
  )
  const [procedures, setProcedures] = useState(initial.planned_procedures.join(', '))
  const [error, setError] = useState<string | null>(null)

  const set = <K extends keyof HealthProfile>(key: K, value: HealthProfile[K]) => setP((x) => ({ ...x, [key]: value }))
  const toggleIn = <T,>(list: T[], item: T) => (list.includes(item) ? list.filter((x) => x !== item) : [...list, item])

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!Number.isInteger(p.age) || p.age < 0 || p.age > 110) return setError('Enter an age between 0 and 110.')
    if (!p.covered_members.length) return setError('Choose at least one person to cover.')
    if (p.height_cm !== null && (p.height_cm < 40 || p.height_cm > 250)) return setError('Height should be between 40 and 250 cm.')
    if (p.weight_kg !== null && (p.weight_kg < 2 || p.weight_kg > 350)) return setError('Weight should be between 2 and 350 kg.')
    if (p.desired_sum_insured !== null && (p.desired_sum_insured < 0 || p.desired_sum_insured > 1e9))
      return setError('Sum insured should be in rupees, up to ₹100 crore (1,000,000,000).')
    if (p.annual_budget !== null && (p.annual_budget < 0 || p.annual_budget > 1e8))
      return setError('Premium budget should be in rupees, up to ₹10 crore (100,000,000).')
    setError(null)
    const common = p.conditions.filter((c) => COMMON_CONDITIONS.includes(c))
    onSubmit({
      ...p,
      conditions: [...common, ...splitList(otherConditions)].slice(0, 20),
      planned_procedures: splitList(procedures).slice(0, 10),
    })
  }

  return (
    <form className="fit-form" onSubmit={submit}>
      <p className="muted small">
        Your answers are used only for this assessment. They are not saved, and they are sent to the language model
        together with the policy excerpts.
      </p>

      <details open className="fit-section">
        <summary>Basics</summary>
        <div className="fit-grid">
          <label>
            Age
            <input type="number" min={0} max={110} value={Number.isNaN(p.age) ? '' : p.age} onChange={(e) => set('age', Number(e.target.value))} required />
          </label>
          <label>
            Gender
            <select value={p.gender} onChange={(e) => set('gender', e.target.value as HealthProfile['gender'])}>
              <option value="unspecified">Prefer not to say</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
              <option value="other">Other</option>
            </select>
          </label>
          <label>
            City
            <select value={p.city_tier} onChange={(e) => set('city_tier', e.target.value as HealthProfile['city_tier'])}>
              <option value="unspecified">Not specified</option>
              <option value="metro">Metro / tier 1</option>
              <option value="tier2">Tier 2</option>
              <option value="tier3">Tier 3 / rural</option>
            </select>
          </label>
          <label>
            Desired sum insured (₹)
            <input type="number" min={0} max={1e9} step={1} value={p.desired_sum_insured ?? ''} placeholder="e.g. 1000000" onChange={(e) => set('desired_sum_insured', numOrNull(e.target.value))} />
            <span className="fit-hint">{inWords(p.desired_sum_insured)}</span>
          </label>
          <label>
            Annual premium budget (₹)
            <input type="number" min={0} max={1e8} step={1} value={p.annual_budget ?? ''} placeholder="e.g. 25000" onChange={(e) => set('annual_budget', numOrNull(e.target.value))} />
            <span className="fit-hint">{inWords(p.annual_budget)}</span>
          </label>
        </div>
        <div className="fit-label">Who should be covered?</div>
        <div className="fit-checks">
          {MEMBERS.map((m) => (
            <label key={m.id} className="check">
              <input type="checkbox" checked={p.covered_members.includes(m.id)} onChange={() => set('covered_members', toggleIn(p.covered_members, m.id))} />
              {m.label}
            </label>
          ))}
        </div>
      </details>

      <details open className="fit-section">
        <summary>Medical history</summary>
        <div className="fit-label">Pre-existing conditions</div>
        <div className="fit-checks">
          {COMMON_CONDITIONS.map((c) => (
            <label key={c} className="check">
              <input type="checkbox" checked={p.conditions.includes(c)} onChange={() => set('conditions', toggleIn(p.conditions, c))} />
              {c}
            </label>
          ))}
        </div>
        <div className="fit-grid">
          <label className="wide">
            Other conditions (comma-separated)
            <input value={otherConditions} maxLength={300} onChange={(e) => setOtherConditions(e.target.value)} placeholder="e.g. PCOS, migraine" />
          </label>
          <label className="wide">
            Current medications
            <input value={p.medications} maxLength={300} onChange={(e) => set('medications', e.target.value)} />
          </label>
          <label className="wide">
            Past surgeries
            <input value={p.past_surgeries} maxLength={300} onChange={(e) => set('past_surgeries', e.target.value)} />
          </label>
        </div>
        <label className="check">
          <input type="checkbox" checked={p.hospitalised_last_2y} onChange={(e) => set('hospitalised_last_2y', e.target.checked)} />
          Hospitalised in the last 2 years
        </label>
      </details>

      <details open className="fit-section">
        <summary>Lifestyle</summary>
        <div className="fit-grid">
          <label>
            Height (cm)
            <input type="number" min={40} max={250} value={p.height_cm ?? ''} onChange={(e) => set('height_cm', numOrNull(e.target.value))} />
          </label>
          <label>
            Weight (kg)
            <input type="number" min={2} max={350} value={p.weight_kg ?? ''} onChange={(e) => set('weight_kg', numOrNull(e.target.value))} />
          </label>
          <label>
            Alcohol
            <select value={p.alcohol} onChange={(e) => set('alcohol', e.target.value as HealthProfile['alcohol'])}>
              <option value="none">None</option>
              <option value="occasional">Occasional</option>
              <option value="regular">Regular</option>
            </select>
          </label>
          <label className="wide">
            Risky occupation or sports
            <input value={p.hazardous_activity} maxLength={300} onChange={(e) => set('hazardous_activity', e.target.value)} placeholder="e.g. trekking, motor racing" />
          </label>
        </div>
        <label className="check">
          <input type="checkbox" checked={p.smoker} onChange={(e) => set('smoker', e.target.checked)} />
          Smoker or tobacco user
        </label>
      </details>

      <details open className="fit-section">
        <summary>Plans and priorities</summary>
        <div className="fit-grid">
          <label className="wide">
            Planned procedures or surgeries (comma-separated)
            <input value={procedures} maxLength={300} onChange={(e) => setProcedures(e.target.value)} placeholder="e.g. knee replacement, cataract" />
          </label>
        </div>
        <div className="fit-checks">
          {(
            [
              ['planning_pregnancy', 'Planning a pregnancy'],
              ['wants_opd', 'Want OPD cover'],
              ['wants_dental', 'Want dental cover'],
              ['wants_ayush', 'Want AYUSH cover'],
              ['avoid_copay', 'Prefer no co-payment'],
              ['avoid_room_rent_limit', 'Prefer no room rent limit'],
            ] as const
          ).map(([key, label]) => (
            <label key={key} className="check">
              <input type="checkbox" checked={p[key]} onChange={(e) => set(key, e.target.checked)} />
              {label}
            </label>
          ))}
        </div>
      </details>

      {(error || serverError) && <div className="error">{error || serverError}</div>}
      <div className="fit-actions">
        <button type="submit" className="primary" disabled={busy}>
          {busy ? 'Scoring…' : 'Score this policy for me'}
        </button>
      </div>
    </form>
  )
}
