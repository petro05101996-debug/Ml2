import React from "react";
export default function InfoCard({ title, children }: { title: string; children: React.ReactNode }) { return <section><h3>{title}</h3>{children}</section>; }
