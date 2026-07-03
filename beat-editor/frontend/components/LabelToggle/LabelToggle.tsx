interface LabelToggleProps {
  isLabelOn: boolean;
  setIsLabelOn: (value: boolean) => void;
}

const LabelToggle = ({ isLabelOn, setIsLabelOn }: LabelToggleProps) => {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={isLabelOn}
      onClick={() => setIsLabelOn(!isLabelOn)}
      className={`inline-flex shrink-0 items-center gap-3 rounded-md border-2 px-3 py-2 text-sm font-bold uppercase tracking-[2px] transition-colors ${
        isLabelOn
          ? "border-transparent bg-[#ee8a78] text-white"
          : "border-transparent bg-[#47555e] text-white hover:bg-[#3d4951]"
      }`}
    >
      <span>Labels</span>
      <span
        className={`relative h-5 w-10 overflow-hidden rounded-full transition-colors ${
          isLabelOn ? "bg-white/30" : "bg-white/20"
        }`}
      >
        <span
          className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${
            isLabelOn ? "translate-x-5" : "translate-x-0"
          }`}
        />
      </span>
    </button>
  );
};

export default LabelToggle;
