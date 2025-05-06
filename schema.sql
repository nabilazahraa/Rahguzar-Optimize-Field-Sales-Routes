--
-- PostgreSQL database dump
--

-- Dumped from database version 17.1
-- Dumped by pg_dump version 17.4 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: app_users; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.app_users (
    appuserid integer NOT NULL,
    appusercode character varying(20)
);


ALTER TABLE public.app_users OWNER TO PostgresAdmin;

--
-- Name: distributor_kpi_history; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.distributor_kpi_history (
    id integer NOT NULL,
    distributorid integer NOT NULL,
    plan_id character varying(50) NOT NULL,
    total_distance_travelled double precision DEFAULT 0,
    avg_distance_travelled double precision DEFAULT 0,
    total_shops_visited integer DEFAULT 0,
    avg_shops_visited double precision DEFAULT 0,
    running_time double precision DEFAULT 0,
    total_workload double precision DEFAULT 0,
    avg_workload double precision DEFAULT 0,
    response_time double precision
);


ALTER TABLE public.distributor_kpi_history OWNER TO PostgresAdmin;

--
-- Name: distributor_kpi_history_id_seq; Type: SEQUENCE; Schema: public; Owner: PostgresAdmin
--

CREATE SEQUENCE public.distributor_kpi_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.distributor_kpi_history_id_seq OWNER TO PostgresAdmin;

--
-- Name: distributor_kpi_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: PostgresAdmin
--

ALTER SEQUENCE public.distributor_kpi_history_id_seq OWNED BY public.distributor_kpi_history.id;


--
-- Name: distributor_stores; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.distributor_stores (
    distributorid integer NOT NULL,
    storeid integer NOT NULL,
    latitude numeric(10,7),
    longitude numeric(10,7)
);


ALTER TABLE public.distributor_stores OWNER TO PostgresAdmin;

--
-- Name: distributors; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.distributors (
    distributorid integer NOT NULL,
    distributorcode character varying(20)
);


ALTER TABLE public.distributors OWNER TO PostgresAdmin;

--
-- Name: graph_total_ob_metrics; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.graph_total_ob_metrics (
    id integer NOT NULL,
    plan_id character varying(50),
    metric_date date NOT NULL,
    total_distance numeric DEFAULT 0,
    total_visits integer DEFAULT 0,
    total_workload numeric DEFAULT 0,
    orderbooker_id integer DEFAULT '-1'::integer NOT NULL
);


ALTER TABLE public.graph_total_ob_metrics OWNER TO PostgresAdmin;

--
-- Name: graph_total_ob_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: PostgresAdmin
--

CREATE SEQUENCE public.graph_total_ob_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.graph_total_ob_metrics_id_seq OWNER TO PostgresAdmin;

--
-- Name: graph_total_ob_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: PostgresAdmin
--

ALTER SEQUENCE public.graph_total_ob_metrics_id_seq OWNED BY public.graph_total_ob_metrics.id;


--
-- Name: graph_unique_ob_metrics; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.graph_unique_ob_metrics (
    id integer NOT NULL,
    plan_id character varying(50) NOT NULL,
    orderbooker_id integer NOT NULL,
    metric_date date NOT NULL,
    distance numeric DEFAULT 0,
    visits integer DEFAULT 0,
    workload numeric DEFAULT 0
);


ALTER TABLE public.graph_unique_ob_metrics OWNER TO PostgresAdmin;

--
-- Name: graph_unique_ob_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: PostgresAdmin
--

CREATE SEQUENCE public.graph_unique_ob_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.graph_unique_ob_metrics_id_seq OWNER TO PostgresAdmin;

--
-- Name: graph_unique_ob_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: PostgresAdmin
--

ALTER SEQUENCE public.graph_unique_ob_metrics_id_seq OWNED BY public.graph_unique_ob_metrics.id;


--
-- Name: orderbooker_kpi_history; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.orderbooker_kpi_history (
    id integer NOT NULL,
    distributorid integer NOT NULL,
    plan_id character varying(50) NOT NULL,
    ob_id integer NOT NULL,
    total_distance_travelled double precision DEFAULT 0,
    avg_distance_travelled double precision DEFAULT 0,
    total_shops_visited integer DEFAULT 0,
    avg_shops_visited double precision DEFAULT 0,
    running_time double precision DEFAULT 0,
    total_workload double precision DEFAULT 0,
    avg_workload double precision DEFAULT 0,
    response_time double precision
);


ALTER TABLE public.orderbooker_kpi_history OWNER TO PostgresAdmin;

--
-- Name: orderbooker_kpi_history_id_seq; Type: SEQUENCE; Schema: public; Owner: PostgresAdmin
--

CREATE SEQUENCE public.orderbooker_kpi_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.orderbooker_kpi_history_id_seq OWNER TO PostgresAdmin;

--
-- Name: orderbooker_kpi_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: PostgresAdmin
--

ALTER SEQUENCE public.orderbooker_kpi_history_id_seq OWNED BY public.orderbooker_kpi_history.id;


--
-- Name: pjp; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.pjp (
    pjpcode character varying(20) NOT NULL,
    distributor_ref integer,
    appuser_ref integer
);


ALTER TABLE public.pjp OWNER TO PostgresAdmin;

--
-- Name: pjp_plans; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.pjp_plans (
    pjp_id character varying(20) NOT NULL,
    plan_id character varying(20),
    orderbooker_id integer NOT NULL,
    plan_date date NOT NULL
);


ALTER TABLE public.pjp_plans OWNER TO PostgresAdmin;

--
-- Name: pjp_store_assignments; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.pjp_store_assignments (
    assignment_id integer NOT NULL,
    pjp_id character varying(20),
    store_id integer NOT NULL,
    visit_sequence integer NOT NULL
);


ALTER TABLE public.pjp_store_assignments OWNER TO PostgresAdmin;

--
-- Name: pjp_store_assignments_assignment_id_seq; Type: SEQUENCE; Schema: public; Owner: PostgresAdmin
--

CREATE SEQUENCE public.pjp_store_assignments_assignment_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.pjp_store_assignments_assignment_id_seq OWNER TO PostgresAdmin;

--
-- Name: pjp_store_assignments_assignment_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: PostgresAdmin
--

ALTER SEQUENCE public.pjp_store_assignments_assignment_id_seq OWNED BY public.pjp_store_assignments.assignment_id;


--
-- Name: plan_master; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.plan_master (
    plan_id character varying(50) NOT NULL,
    distributor_id integer NOT NULL,
    plan_duration integer NOT NULL,
    created_date timestamp without time zone NOT NULL,
    status boolean DEFAULT true NOT NULL
);


ALTER TABLE public.plan_master OWNER TO PostgresAdmin;

--
-- Name: plan_master_ob_days; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.plan_master_ob_days (
    id integer NOT NULL,
    plan_id character varying(20) NOT NULL,
    orderbooker_id integer NOT NULL,
    num_days integer NOT NULL
);


ALTER TABLE public.plan_master_ob_days OWNER TO PostgresAdmin;

--
-- Name: plan_master_ob_days_id_seq; Type: SEQUENCE; Schema: public; Owner: PostgresAdmin
--

CREATE SEQUENCE public.plan_master_ob_days_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.plan_master_ob_days_id_seq OWNER TO PostgresAdmin;

--
-- Name: plan_master_ob_days_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: PostgresAdmin
--

ALTER SEQUENCE public.plan_master_ob_days_id_seq OWNED BY public.plan_master_ob_days.id;


--
-- Name: secondary_sales; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.secondary_sales (
    distributorid integer,
    appuserid integer,
    storeid integer NOT NULL,
    invoicenumber character varying(50),
    fiscaldate date NOT NULL,
    netsales double precision,
    salesunits integer,
    tp_inc_gst double precision
);


ALTER TABLE public.secondary_sales OWNER TO PostgresAdmin;

--
-- Name: store_channel; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.store_channel (
    storeid integer NOT NULL,
    channeltypeid smallint,
    channelid smallint,
    subchannelid smallint
);


ALTER TABLE public.store_channel OWNER TO PostgresAdmin;

--
-- Name: store_classification; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.store_classification (
    storeid integer NOT NULL,
    storeclassificationoneid smallint,
    storeclassificationtwoid smallint,
    storeclassificationthreeid smallint
);


ALTER TABLE public.store_classification OWNER TO PostgresAdmin;

--
-- Name: store_hierarchy; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.store_hierarchy (
    storeid integer NOT NULL,
    storecode character varying(20),
    storeregistered boolean,
    status smallint,
    storeperfectid smallint,
    storetype smallint,
    storefilertype smallint
);


ALTER TABLE public.store_hierarchy OWNER TO PostgresAdmin;

--
-- Name: store_location; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.store_location (
    storeid integer NOT NULL,
    areatype smallint,
    townid integer,
    localityid integer,
    sublocalityid integer,
    latitude numeric(10,7),
    longitude numeric(10,7)
);


ALTER TABLE public.store_location OWNER TO PostgresAdmin;

--
-- Name: universe_stores; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.universe_stores (
    storecode character varying(20),
    sublocalityid integer,
    storestatus character varying(10),
    distributor_ref integer,
    appuser_ref integer,
    pjp_ref character varying(20)
);


ALTER TABLE public.universe_stores OWNER TO PostgresAdmin;

--
-- Name: users; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.users (
    distributor_id integer NOT NULL,
    username character varying(50) NOT NULL,
    password_hash text NOT NULL,
    email character varying(100) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.users OWNER TO PostgresAdmin;

--
-- Name: visits; Type: TABLE; Schema: public; Owner: PostgresAdmin
--

CREATE TABLE public.visits (
    visitid integer NOT NULL,
    storeid integer,
    invoicestatus smallint,
    visitcomplete boolean,
    closereasonid smallint,
    latitude numeric(10,7),
    longitude numeric(10,7),
    syncdown smallint,
    syncup timestamp without time zone,
    syncdowndatetime timestamp without time zone,
    visitstarttime timestamp without time zone,
    visitendtime timestamp without time zone,
    visitspenttimeinseconds integer,
    distanceinmeters numeric(10,2),
    distributor_ref integer,
    appuser_ref integer,
    pjp_ref character varying(20)
);


ALTER TABLE public.visits OWNER TO PostgresAdmin;

--
-- Name: distributor_kpi_history id; Type: DEFAULT; Schema: public; Owner: PostgresAdmin
--

ALTER TABLE ONLY public.distributor_kpi_history ALTER COLUMN id SET DEFAULT nextval('public.distributor_kpi_history_id_seq'::regclass);


--
-- Name: graph_total_ob_metrics id; Type: DEFAULT; Schema: public; Owner: PostgresAdmin
--

ALTER TABLE ONLY public.graph_total_ob_metrics ALTER COLUMN id SET DEFAULT nextval('public.graph_total_ob_metrics_id_seq'::regclass);


--
-- Name: graph_unique_ob_metrics id; Type: DEFAULT; Schema: public; Owner: PostgresAdmin
--

ALTER TABLE ONLY public.graph_unique_ob_metrics ALTER COLUMN id SET DEFAULT nextval('public.graph_unique_ob_metrics_id_seq'::regclass);


--
-- Name: orderbooker_kpi_history id; Type: DEFAULT; Schema: public; Owner: PostgresAdmin
--

ALTER TABLE ONLY public.orderbooker_kpi_history ALTER COLUMN id SET DEFAULT nextval('public.orderbooker_kpi_history_id_seq'::regclass);


--
-- Name: pjp_store_assignments assignment_id; Type: DEFAULT; Schema: public; Owner: PostgresAdmin
--

ALTER TABLE ONLY public.pjp_store_assignments ALTER COLUMN assignment_id SET DEFAULT nextval('public.pjp_store_assignments_assignment_id_seq'::regclass);


--
-- Name: plan_master_ob_days id; Type: DEFAULT; Schema: public; Owner: PostgresAdmin
--

ALTER TABLE ONLY public.plan_master_ob_days ALTER COLUMN id SET DEFAULT nextval('public.plan_master_ob_days_id_seq'::regclass);

