import type { Toast as ToastType } from '../types';

interface ToastProps {
  toast: ToastType;
  onClose: (id: number) => void;
}

export const Toast = ({ toast, onClose }: ToastProps) => {
  const getIcon = () => {
    switch (toast.type) {
      case 'error':
        return '⚠️';
      case 'success':
        return '✓';
      case 'warning':
        return '⚠️';
      default:
        return 'ℹ️';
    }
  };

  return (
    <div className={`toast ${toast.type}`}>
      <span className="toast-icon">{getIcon()}</span>
      <span className="toast-content">{toast.message}</span>
      <button className="toast-close" onClick={() => onClose(toast.id)}>
        ×
      </button>
    </div>
  );
};

interface ToastContainerProps {
  toasts: ToastType[];
  onClose: (id: number) => void;
}

export const ToastContainer = ({ toasts, onClose }: ToastContainerProps) => {
  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <Toast key={toast.id} toast={toast} onClose={onClose} />
      ))}
    </div>
  );
};

