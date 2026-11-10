import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'

export default function PatientList() {
  const [patients, setPatients] = useState([])
  const [loading, setLoading]   = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    getPatients()
      .then(r => setPatients(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-3xl mx-auto p-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-semibold text-gray-800">Patients</h1>
        <span className="text-sm text-gray-400">{patients.length} registered</span>
      </div>

      {loading && <p className="text-gray-400">Loading...</p>}

      <div className="space-y-3">
        {patients.map(p => (
          <div key={p.id}
            className="bg-white rounded-xl border border-gray-200 px-6 py-4 flex items-center justify-between hover:shadow-md transition cursor-pointer"
            onClick={() => navigate(`/physio/${p.id}/history`)}
          >
            <div>
              <p className="font-medium text-gray-800">{p.name}</p>
              <p className="text-sm text-gray-400">ID #{p.id}  {p.notes || ''}</p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={e => { e.stopPropagation(); navigate(`/physio/${p.id}/live`) }}
                className="text-sm bg-green-100 text-green-700 px-3 py-1 rounded-lg hover:bg-green-200 transition"
              >
                Live
              </button>
              <button
                onClick={e => { e.stopPropagation(); navigate(`/physio/${p.id}/history`) }}
                className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-lg hover:bg-blue-200 transition"
              >
                History
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}