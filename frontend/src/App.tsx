import { useEffect, useRef } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useAuth } from "./context/AuthContext";
import { api } from "./api/client";
import { SessionWatcher } from "./components/SessionWatcher";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard/Dashboard";
import PatientList from "./pages/Dashboard/PatientList";
import PatientDetail from "./pages/Dashboard/PatientDetail";
import ApiAccess from "./pages/Admin/ApiAccess";
import Team from "./pages/Admin/Team";
import Hospitals from "./pages/Admin/Hospitals";
import HospitalDetail from "./pages/Admin/HospitalDetail";
import QuestionBank from "./pages/Admin/QuestionBank";
import AuditLog from "./pages/Admin/AuditLog";
import PlatformUsage from "./pages/Admin/PlatformUsage";
import MyCases from "./pages/Doctor/MyCases";
import ReportView from "./pages/Reports/ReportView";
import CombinedReportView from "./pages/Reports/CombinedReportView";
import ChangePassword from "./pages/Account/ChangePassword";
import { DiagnosisFlowProvider } from "./pages/NewDiagnosis/DiagnosisFlowContext";
import PatientSelect from "./pages/NewDiagnosis/PatientSelect";
import ChooseType from "./pages/NewDiagnosis/ChooseType";
import Questionnaire from "./pages/NewDiagnosis/Questionnaire";
import PersonalInfo from "./pages/NewDiagnosis/PersonalInfo";
import ImageUpload from "./pages/NewDiagnosis/ImageUpload";
import Results from "./pages/NewDiagnosis/Results";

function RequireAuth({ children }: { children: JSX.Element }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function RequireRole({ roles, children }: { roles: string[]; children: JSX.Element }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return roles.includes(user.role) ? children : <Navigate to="/dashboard" replace />;
}

function PageViewTracker() {
  const { isAuthenticated } = useAuth();
  const location = useLocation();
  const lastPath = useRef<string | null>(null);

  useEffect(() => {
    if (!isAuthenticated) return;
    if (lastPath.current === location.pathname) return;
    lastPath.current = location.pathname;
    api.logPageView(location.pathname).catch(() => {});
  }, [isAuthenticated, location.pathname]);

  return null;
}

export default function App() {
  return (
    <>
      <PageViewTracker />
      <SessionWatcher />
      <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />

      <Route
        path="/dashboard"
        element={
          <RequireAuth>
            <Dashboard />
          </RequireAuth>
        }
      />
      <Route
        path="/patients"
        element={
          <RequireRole roles={["staff", "hospital_admin"]}>
            <PatientList />
          </RequireRole>
        }
      />
      <Route
        path="/patients/:patientId"
        element={
          <RequireRole roles={["staff", "hospital_admin"]}>
            <PatientDetail />
          </RequireRole>
        }
      />
      <Route
        path="/cases"
        element={
          <RequireRole roles={["doctor"]}>
            <MyCases />
          </RequireRole>
        }
      />

      <Route
        path="/reports/:sessionId"
        element={
          <RequireAuth>
            <ReportView />
          </RequireAuth>
        }
      />
      <Route
        path="/combined-reports/:reportId"
        element={
          <RequireAuth>
            <CombinedReportView />
          </RequireAuth>
        }
      />
      <Route
        path="/account"
        element={
          <RequireAuth>
            <ChangePassword />
          </RequireAuth>
        }
      />

      <Route
        path="/admin/api-access"
        element={
          <RequireRole roles={["hospital_admin"]}>
            <ApiAccess />
          </RequireRole>
        }
      />
      <Route
        path="/admin/team"
        element={
          <RequireRole roles={["hospital_admin"]}>
            <Team />
          </RequireRole>
        }
      />
      <Route
        path="/platform/hospitals"
        element={
          <RequireRole roles={["platform_admin"]}>
            <Hospitals />
          </RequireRole>
        }
      />
      <Route
        path="/platform/hospitals/:hospitalId"
        element={
          <RequireRole roles={["platform_admin"]}>
            <HospitalDetail />
          </RequireRole>
        }
      />
      <Route
        path="/platform/questions"
        element={
          <RequireRole roles={["platform_admin"]}>
            <QuestionBank />
          </RequireRole>
        }
      />
      <Route
        path="/platform/audit-log"
        element={
          <RequireRole roles={["platform_admin"]}>
            <AuditLog />
          </RequireRole>
        }
      />
      <Route
        path="/platform/usage"
        element={
          <RequireRole roles={["platform_admin"]}>
            <PlatformUsage />
          </RequireRole>
        }
      />

      <Route
        path="/diagnose/*"
        element={
          <RequireRole roles={["staff", "hospital_admin", "doctor"]}>
            <DiagnosisFlowProvider>
              <Routes>
                <Route index element={<PatientSelect />} />
                <Route path="type" element={<ChooseType />} />
                <Route path="questionnaire" element={<Questionnaire />} />
                <Route path="personal-info" element={<PersonalInfo />} />
                <Route path="images" element={<ImageUpload />} />
                <Route path="results" element={<Results />} />
              </Routes>
            </DiagnosisFlowProvider>
          </RequireRole>
        }
      />
      </Routes>
    </>
  );
}
