function StatusBanner({ status }) {
  if (!status) {
    return null;
  }

  return (
    <div className={`banner ${status.type}`}>
      <span>{status.message}</span>
    </div>
  );
}

export default StatusBanner;
