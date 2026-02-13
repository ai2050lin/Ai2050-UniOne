import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Activity, AlertCircle, TrendingUp, Users, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const TrainingMonitor = () => {
  const [agents, setAgents] = useState([]);
  const [selectedAgentId, setSelectedAgentId] = useState(null);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/training/status');
      const result = await response.json();
      
      if (result.status === 'running') {
        // Handle new multi-agent response format
        // Structure: { status: "running", agents: [ { id: "name", last_updated: float, data: {...} } ] }
        // Or legacy: { status: "running", data: {...}, last_updated: float }
        
        let agentList = [];
        if (result.agents) {
            agentList = result.agents;
        } else if (result.data) {
            // Legacy/Single fallback
            agentList = [{
                id: "default",
                last_updated: result.last_updated,
                data: result.data
            }];
        }

        if (agentList.length === 0) {
            setStatus('no_training');
            return;
        }

        // Update agents list
        setAgents(agentList);
        setStatus('running');
        setError(null);
        
        // Auto-select first agent if none selected or selected one disappeared
        if (!selectedAgentId || !agentList.find(a => a.id === selectedAgentId)) {
            if (agentList.length > 0) {
                setSelectedAgentId(agentList[0].id);
            }
        }
      } else {
        setStatus(result.status);
        if (result.message) setError(result.message);
      }
    } catch (err) {
      console.error("Failed to fetch training status:", err);
      setStatus('error');
      setError("Failed to connect to backend.");
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000); // Poll every second
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Get selected agent data
  const selectedAgent = agents.find(a => a.id === selectedAgentId);
  const trainingData = selectedAgent ? (() => {
      // Process data for charts
      const transformerLogs = selectedAgent.data.Transformer || [];
      const fiberNetLogs = selectedAgent.data.FiberNet || [];
      const maxLength = Math.max(transformerLogs.length, fiberNetLogs.length);
      const merged = [];
      for (let i = 0; i < maxLength; i++) {
           const tLog = transformerLogs[i] || {};
           const fLog = fiberNetLogs[i] || {};
           merged.push({
             epoch: tLog.epoch !== undefined ? tLog.epoch : fLog.epoch,
             trans_acc: tLog.accuracy,
             trans_loss: tLog.loss,
             fiber_acc: fLog.accuracy,
             fiber_curv: fLog.curvature
           });
      }
      return merged;
  })() : [];

  const lastUpdatedTime = selectedAgent && selectedAgent.last_updated 
        ? new Date(selectedAgent.last_updated * 1000).toLocaleTimeString() 
        : '-';

  if (status === 'loading') {
    return <div className="p-4 text-center text-gray-400">Loading training agents...</div>;
  }

  if (status === 'no_training' || status === 'error' && agents.length === 0) {
    return (
      <Card className="w-full bg-slate-900 border-slate-700 text-white mt-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Activity className="h-5 w-5 text-gray-400" />
            Training Monitor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center p-12 text-gray-400 border border-dashed border-gray-700 rounded-lg">
            <AlertCircle className="h-12 w-12 mb-4 text-gray-600" />
            <p className="text-lg font-semibold text-gray-300">{error || "No active training agents found."}</p>
            <p className="text-sm mt-2">Start a training script to see real-time metrics.</p>
            <code className="mt-4 bg-slate-800 p-2 rounded text-xs text-blue-400">python scripts/run_toy_training.py --name Agent1</code>
          </div>
        </CardContent>
      </Card>
    );
  }

  const latest = trainingData.length > 0 ? trainingData[trainingData.length - 1] : null;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Agent Selector */}
      <div className="flex items-center justify-between bg-slate-800 p-4 rounded-lg border border-slate-700">
        <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-white font-medium">
                <Users className="h-5 w-5 text-indigo-400" />
                <span>Active Agents: {agents.length}</span>
            </div>
            <Select value={selectedAgentId} onValueChange={setSelectedAgentId}>
            <SelectTrigger className="w-[200px] bg-slate-900 border-slate-700 text-white">
                <SelectValue placeholder="Select Agent" />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-slate-700 text-white">
                {agents.map(agent => (
                <SelectItem key={agent.id} value={agent.id} className="focus:bg-slate-800 cursor-pointer">
                    {agent.id}
                </SelectItem>
                ))}
            </SelectContent>
            </Select>
        </div>
        <div className="text-sm text-gray-400">
            Last update: <span className="text-gray-200">{lastUpdatedTime}</span>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-slate-800 border-slate-700 shadow-lg">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-400 flex items-center gap-2">
              <Zap className="h-4 w-4 text-yellow-500" />
              Current Epoch
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white tracking-tight">{latest ? latest.epoch : '-'}</div>
            <div className="text-xs text-gray-500 mt-1">Simulated Time</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700 shadow-lg">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-400 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-blue-500" />
              Transformer Acc
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-400 tracking-tight">{latest && latest.trans_acc ? latest.trans_acc.toFixed(2) : '-'}%</div>
            <div className="text-xs text-gray-500 mt-1">Stochastic Baseline</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700 shadow-lg">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-400 flex items-center gap-2">
              <Activity className="h-4 w-4 text-green-500" />
              FiberNet Acc
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-400 tracking-tight">{latest && latest.fiber_acc ? latest.fiber_acc.toFixed(2) : '-'}%</div>
            <div className="text-xs text-gray-500 mt-1">Geometric Optimization</div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-slate-800 border-slate-700 shadow-xl overflow-hidden">
          <CardHeader className="bg-slate-800/50 border-b border-slate-700/50">
            <CardTitle className="text-white text-sm font-medium">Accuracy Comparison ({selectedAgentId})</CardTitle>
          </CardHeader>
          <CardContent className="h-[320px] p-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" domain={[0, 100]} fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '12px' }}
                  labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
                />
                <Legend iconType="circle" />
                <Line type="monotone" dataKey="trans_acc" name="Transformer" stroke="#60a5fa" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="fiber_acc" name="FiberNet" stroke="#4ade80" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700 shadow-xl overflow-hidden">
          <CardHeader className="bg-slate-800/50 border-b border-slate-700/50">
            <CardTitle className="text-white text-sm font-medium">Curvature vs Loss ({selectedAgentId})</CardTitle>
          </CardHeader>
          <CardContent className="h-[320px] p-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis yAxisId="left" stroke="#ef4444" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '12px' }}
                />
                <Legend iconType="circle" />
                <Line yAxisId="left" type="monotone" dataKey="fiber_curv" name="Fiber Curvature" stroke="#ef4444" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
                <Line yAxisId="right" type="monotone" dataKey="trans_loss" name="Transformer Loss" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default TrainingMonitor;
